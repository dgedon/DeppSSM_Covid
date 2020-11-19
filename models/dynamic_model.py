import torch.nn as nn
from . import STORN


class DynamicModel(nn.Module):
    def __init__(self, model, num_inputs, num_outputs, options, normalizer_input=None, normalizer_output=None,
                 *args, **kwargs):
        super(DynamicModel, self).__init__()
        # Save parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.args = args
        self.kwargs = kwargs
        self.normalizer_input = normalizer_input
        self.normalizer_output = normalizer_output
        self.zero_initial_state = False

        model_options = options['model_options']

        # initialize the model
        if model == 'STORN':
            self.m = STORN(model_options, options['device'])
        else:
            raise Exception("Unimplemented model!")

    @property
    def num_model_inputs(self):
        return self.num_inputs + self.num_outputs if self.ar else self.num_inputs

    def forward(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)
        if y is not None and self.normalizer_output is not None:
            y = self.normalizer_output.normalize(y)

        loss = self.m(u, y)

        return loss

    def generate(self, u, y=None):
        if self.normalizer_input is not None:
            u = self.normalizer_input.normalize(u)

        y_sample, y_sample_mu, y_sample_sigma, _ = self.m.generate(u)

        if self.normalizer_output is not None:
            y_sample = self.normalizer_output.unnormalize(y_sample)
        if self.normalizer_output is not None:
            y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)
        if self.normalizer_output is not None:
            y_sample_sigma = self.normalizer_output.unnormalize_sigma(y_sample_sigma)

        return y_sample, y_sample_mu, y_sample_sigma

    def generate_recursively(self, u, T, N):
        """
        Use 1-step ahead predictor until step T
        Then switch for N steps to recursive prediction
        """

        # only take u for the values of the 1-step ahead predictor
        u_truncated = u[:, :, :T]

        if self.normalizer_input is not None:
            u_truncated = self.normalizer_input.normalize(u_truncated)

        y_sample_mu = self.m.generate_recursively(u_truncated, N)

        if self.normalizer_output is not None:
            y_sample_mu = self.normalizer_output.unnormalize_mean(y_sample_mu)

        return y_sample_mu
