import numpy as np
from statsmodels.tsa.statespace.mlemodel import MLEModel


class SemiLinearTrend(MLEModel):
    def __init__(self, endog):
        """
        Univariate Local Linear Trend Model
        """
        # Model order
        k_states = k_posdef = 2

        # Initialize the statespace
        super(SemiLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states)
        
        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend', 'rho']

    @property
    def start_params(self):
        return np.r_[[np.nanstd(self.endog)]*3, 0.99]

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(SemiLinearTrend, self).update(params, *args, **kwargs)

        # Observation covariance
        self.ssm['obs_cov',0,0] = params[0]

        # State covariance
        self.ssm[self._state_cov_idx] = params[1:3]
        self['transition', 1, 1] = params[3]


class LocalLinearTrend(MLEModel):
    def __init__(self, endog, obs_cov=None):
        """
        Univariate Local Linear Trend Model
        """
        # Model order
        k_states = k_posdef = 2
        if obs_cov is not None:
            self.obs_cov_median = np.median(obs_cov)
            obs_cov = obs_cov[np.newaxis, np.newaxis, :]
           
            
        # Initialize the statespace
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )
        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])
        self.ssm['selection'] = np.eye(k_states)

        # Cache some indices
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)
        if obs_cov is not None:
            self.ssm['obs_cov'] = obs_cov
        self.obs_cov = obs_cov
        
    def clone(self, endog, exog, **kwargs):
        # This method must be set to allow forecasting in custom
        # state space models that include time-varying state
        # space matrices, like we have for state_intercept here
        obs_cov = np.ones(endog.shape[0])*self.obs_cov_median
        mod = self.__class__(endog, obs_cov=obs_cov)
        return mod
    
    @property
    def param_names(self):
        if self.obs_cov is None:
            return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']
        return ['sigma2.level', 'sigma2.trend']

    @property
    def start_params(self):
        if self.obs_cov is None:
            return [np.nanstd(self.endog)]*3
        return [np.nanstd(self.endog)]*2

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        
        if self.obs_cov is None:
            # Observation covariance
            self.ssm['obs_cov',0,0] = params[0]

            # State covariance
            self.ssm[self._state_cov_idx] = params[1:3]
        else:
            # State covariance
            self.ssm[self._state_cov_idx] = params[:]


class SO2FusionModel(MLEModel):
    def __init__(self, measurements, initial_state=0, initial_cov=1,
                 k_states=2, k_posdef=2, obs_cov=None):
        # Initialize the base MLEModel
        super().__init__(endog=measurements, k_states=k_states,
                         k_posdef=k_posdef)
        self.obs_cov = obs_cov
        # Initial state and covariance
        self.ssm.initialize_known(np.array(initial_state),
                                  np.eye(k_states)*initial_cov)

        # Define the state transition matrix (A)

        if k_states == 2:
            self.ssm['transition'] = np.array([[1, 1],
                                               [0, 1]])
        else:
            self.ssm.transition = np.array([[1]])

        # Define the selection matrix (B, for process noise)
        if k_states == 2:
            self.ssm['selection'] = np.eye(k_states)
        else:
            self.ssm.selection = np.array([[1]])

        # Define the measurement matrix (H)
        n_sensors = measurements.shape[1]
        if k_states == 2:
            self.ssm.design = np.tile([1, 0], (n_sensors, 1))
        else:
            self.ssm.design = np.tile([1], (n_sensors, 1))

        if self.obs_cov is not None:
            self.ssm.obs_cov = obs_cov

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5


    def update(self, params, *args, **kwargs):
        # Update the process noise covariance (Q)
        if self.ssm.k_states == 2:
            self.ssm.state_cov = np.diag(params[:2])
        else:
            self.ssm.state_cov = np.array([params[0]])
        # Update the measurement noise covariance (R)
        if self.obs_cov is None:
            self.ssm.obs_cov = np.diag(params[2:])
