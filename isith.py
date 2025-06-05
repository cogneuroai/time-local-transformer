from datetime import datetime

import torch
from math import factorial, log

# Impulse-based SITH class
class iSITH(torch.nn.Module):
    def __init__(self, tau_min=.1, tau_max=100., buff_max=None, k=50, ntau=32, dt=1, g=0.0,
                 ttype=torch.float, group_conv=False, delta_pulse=False, gaussian=False, F=False, device='cpu'):
        super(iSITH, self).__init__()
        """A SITH module using the perfect equation for the resulting ftilde

        Parameters
        ----------

            - tau_min: float
                The center of the temporal receptive field for the first taustar produced. 
            - tau_max: float
                The center of the temporal receptive field for the last taustar produced. 
            - buff_max: int
                The maximum time in which the filters go into the past. NOTE: In order to 
                achieve as few edge effects as possible, buff_max needs to be bigger than
                tau_max, and dependent on k, such that the filters have enough time to reach 
                very close to 0.0. Plot the filters and you will see them go to 0. 
            - k: int
                Temporal Specificity of the taustars. If this number is high, then taustars
                will always be more narrow.
            - ntau: int
                Number of taustars produced, spread out logarithmically.
            - dt: float
                The time delta of the model. The there will be int(buff_max/dt) filters per
                taustar. Essentially this is the base rate of information being presented to the model
            - g: float
                Typically between 0 and 1. This parameter is the scaling factor of the output
                of the module. If set to 1, the output amplitude for a delta function will be
                identical through time. If set to 0, the amplitude will decay into the past, 
                getting smaller and smaller. This value should be picked on an application to 
                application basis.
            - ttype: Torch Tensor
                This is the type we set the internal mechanism of the model to before running. 
                In order to calculate the filters, we must use a DoubleTensor, but this is no 
                longer necessary after they are calculated. By default we set the filters to 
                be FloatTensors. NOTE: If you plan to use CUDA, you need to pass in a 
                cuda.FloatTensor as the ttype, as using .cuda() will not put these filters on 
                the gpu. 


        """
        self.delta_pulse = delta_pulse
        self.gaussian = gaussian
        self.F = F
        self.device = device
        self.k = k
        self.tau_min = tau_min
        self.tau_max = tau_max
        if buff_max is None:
            buff_max = 3 * tau_max
        self.buff_max = buff_max
        self.ntau = ntau
        self.dt = dt
        self.g = g

        self.c = (tau_max / tau_min) ** (1. / (ntau - 1)) - 1

        self.tau_star = tau_min * (1 + self.c) ** torch.arange(ntau).type(torch.DoubleTensor).to(device)

        self.times = torch.arange(dt, buff_max + dt, dt).type(torch.DoubleTensor).to(device)

        a = log(k) * k
        b = torch.log(torch.arange(2, k).type(torch.DoubleTensor)).to(device).sum()

        # A = ((1/self.tau_star)*(k**(k+1)/factorial(k))*(self.tau_star**self.g)).unsqueeze(1)
        A = ((1 / self.tau_star) * (torch.exp(a - b)) * (self.tau_star ** self.g)).unsqueeze(1)

        #delta pulse here
        # 32 * 3 * tau_max
        # first delta pulse 1
        # second delta pulse 2

        if self.delta_pulse:
            # Generate delta pulse filters
            self.filters = torch.zeros(ntau, 1, 1, int(buff_max/dt), device=device, dtype=ttype)
            for idx in range(ntau):
                self.filters[idx, ..., idx] = 1.0
                ## for testing
                #self.filters[idx, ..., idx] = idx + 1
            self.filters = torch.flip(self.filters, [-1])
            self.filters = self.filters.type(ttype).to(device)
        
        elif self.gaussian:
            sigma = 0.5 * self.buff_max / self.ntau  # Fixed sigma
            # Ensure arguments to zeros are integers
            filters = torch.zeros(self.ntau, 1, 1, int(self.buff_max), device=self.device, dtype=ttype)

            # Calculate centers such that all filters are well within the buffer range
            padding = int(sigma * 3)  # Assuming 3 sigma to nearly encompass all the Gaussian data
            effective_buff_max = self.buff_max - 2 * padding
            # Ensure linspace arguments are integers
            centers = torch.linspace(int(padding), int(self.buff_max - padding - 1), int(self.ntau), device=self.device)

            for i in range(self.ntau):
                center = centers[i]
                # Ensure linspace arguments are integers
                t = torch.linspace(0, int(self.buff_max - 1), int(self.buff_max), device=self.device)
                filter_values = torch.exp(-0.5 * ((t - center) / sigma) ** 2)
                filter_values /= filter_values.sum()  # Normalize
                filters[i, 0, 0, :] = filter_values

            filters = torch.flip(filters, dims=[-1])
            self.filters = filters
        elif self.F:
            times_reshaped = self.times.unsqueeze(0) # Shape: [1, num_times] 
            tau_star_reshaped = self.tau_star.unsqueeze(1) # Shape: [ntau, 1]
            exponent = -times_reshaped / tau_star_reshaped
            self.filters = torch.exp(exponent) # Shape: [ntau, num_times]
            self.filters = (1.0 / tau_star_reshaped) * self.filters
            self.filters = self.filters.type(ttype).to(device)
        else: 
            self.filters = A * torch.exp((torch.log(self.times.unsqueeze(0) / self.tau_star.unsqueeze(1)) * (k + 1)) + \
                                        (k * (-self.times.unsqueeze(0) / self.tau_star.unsqueeze(1))))

            self.filters = torch.flip(self.filters, [-1]).unsqueeze(1).unsqueeze(1)
            self.filters = self.filters.type(ttype).to(device)

        self.filters = torch.flip(self.filters, [0]) # make indices make sense, idx of 0 is first on the left
        self.group_conv = group_conv

    def extra_repr(self):
        s = "ntau={ntau}, tau_min={tau_min}, tau_max={tau_max}, buff_max={buff_max}, dt={dt}, k={k}, g={g}"
        s = s.format(**self.__dict__)
        return s

    def forward(self, inp):
        """Takes in (Batch, 1, features, sequence) and returns (Batch, Taustar, features, sequence)"""
        #print(self.tau_max)
        #print(self.ntau)
        assert (len(inp.shape) >= 4)
        batch, ntau, features, sequence = inp.shape
        #if self.group_conv:
            #out = torch.conv2d(inp, self.filters[:, :, :, -inp.shape[-1]:], 
                            #padding=[0, self.filters[:, :, :, -inp.shape[-1]:].shape[-1]], groups=self.filters.shape[0])
        if self.group_conv:
            out = torch.conv2d(inp, self.filters[:, :, :, -inp.shape[-1]:], 
                            padding=[0, self.filters[:, :, :, -inp.shape[-1]:].shape[-1]], groups=ntau)
        else:
            out = torch.conv2d(inp, self.filters[:, :, :, -inp.shape[-1]:],
                            padding=[0, self.filters[:, :, :, -inp.shape[-1]:].shape[-1]])
        # padding=[0, self.filters.shape[-1]])
        # note we're scaling the output by both dt and the k/(k+1)
        # Off by 1 introduced by the conv2d
        return out[:, :, :, 1:inp.shape[-1] + 1]# * self.dt * self.k / (self.k + 1)

# import matplotlib.pyplot as plt
#
# sith_filters = 50
# tau_min = 10
# cap_len = 100
# k = 10
# sith_params = [{
#     "ntau": sith_filters,
#     "tau_min": tau_min * 1e-3,
#     "tau_max": 1e-3 * (cap_len//3),
#     "dt": 0.001,
#     "k": k,
#     "g": 0.,
#     "ttype": torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# }]
#
# sith = iSITH(**sith_params[0])
#
# for f in sith.filters:
#     plt.plot(f.view(-1).numpy())
#
# plt.show()
# print("Done")

