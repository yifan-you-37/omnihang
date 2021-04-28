import numpy as np
from copy import deepcopy

class sepCEM:

	"""
	Cross-entropy methods.
	"""

	def __init__(self, num_params,
				batch_size,
				mu_init=None,
				sigma_init=1e-3,
				sigma_init_aa=1e-3,
				pop_size=256,
				damp=1e-3,
				damp_limit=1e-5,
				damp_aa=0.2,
				damp_limit_aa=0.1,
				parents=None,
				elitism=False,
				antithetic=False):

		# misc
		self.num_params = num_params
		self.batch_size = batch_size

		# distribution parameters
		if mu_init is None:
			self.mu = np.zeros((self.batch_size, self.num_params))
		else:
			self.mu = np.array(mu_init)
		self.sigma = sigma_init
		self.sigma_aa = sigma_init_aa
		self.damp = damp
		self.damp_limit = damp_limit
		self.damp_aa = damp_aa
		self.damp_limit_aa = damp_limit_aa

		self.tau = 0.95
		self.cov = np.ones((self.batch_size, self.num_params))
		self.cov[:, :3] *= self.sigma 
		self.cov[:, 3:] *= self.sigma_aa 

		# elite stuff
		self.elitism = elitism
		self.elite = np.random.rand(self.batch_size, self.num_params)
		self.elite[:, :3] = np.sqrt(self.sigma)
		self.elite[:, 3:] = np.sqrt(self.sigma_aa)

		self.elite_score = None

		# sampling stuff
		self.pop_size = pop_size
		self.antithetic = antithetic

		if self.antithetic:
			assert (self.pop_size % 2 == 0), "Population size must be even"
		if parents is None or parents <= 0:
			self.parents = pop_size // 2
		else:
			self.parents = parents
		self.weights = np.array([np.log((self.parents + 1) / i)
								 for i in range(1, self.parents + 1)])
		self.weights /= self.weights.sum()

	def ask(self, pop_size):
		"""
		Returns a list of candidates parameters
		"""
		if self.antithetic and not pop_size % 2:
			assert False, 'this part not implemented yet'
			epsilon_half = np.random.randn(self.batch_size, pop_size // 2, self.num_params)
			epsilon = np.concatenate([epsilon_half, - epsilon_half])

		else:
			epsilon = np.random.randn(self.batch_size, pop_size, self.num_params)

		# print('mu size', self.mu.shape, 'cov', self.cov[:, np.newaxis, :].shape)
		inds = self.mu[:, np.newaxis, :] + epsilon * np.sqrt(self.cov[:, np.newaxis, :])
		if self.elitism:
			inds[:, -1] = self.elite

		return inds

	def tell(self, solutions, scores):
		"""
		Updates the distribution
		"""
		scores = np.array(scores)
		scores *= -1
		if len(scores.shape) == 1:
			scores = scores[np.newaxis, :]
		idx_sorted = np.argsort(scores, axis=1)

		old_mu = self.mu
		self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
		self.damp_aa = self.damp_aa * self.tau + (1 - self.tau) * self.damp_limit_aa
		idx_sorted = idx_sorted[:, :self.parents]
		top_solutions = np.take_along_axis(solutions, idx_sorted[:, :, np.newaxis], axis=1)
		# print('top solutions', top_solutions.shape, 'weights', self.weights.shape)
		self.mu = self.weights @ top_solutions
		
		# print('new mu', self.mu.shape)
		# mu_tmp = np.zeros_like(self.mu)
		# for i in range(mu_tmp.shape[0]):
		# 	mu_tmp[i] = self.weights @ top_solutions[i]
		# print(np.allclose(mu_tmp, self.mu))

		z = top_solutions - old_mu[:, np.newaxis, :]
		# print('z', z.shape)
		self.cov = 1 / self.parents * self.weights @ (
			z * z) 
		self.cov[:, :3] += self.damp * np.ones((self.batch_size, 3))
		self.cov[:, 3:] += self.damp_aa * np.ones((self.batch_size, 3))

		self.elite = top_solutions[:, 0, :]
		# self.elite_score = scores[idx_sorted[0]]
		# print('cov', self.cov)
		
		return self.elite

	def get_distrib_params(self):
		"""
		Returns the parameters of the distrubtion:
		the mean and sigma
		"""
		return np.copy(self.mu), np.copy(self.cov)

class Searcher():
	def __init__(self,
				action_dim,
				max_action,
				max_action_aa,
				sigma_init=1e-3,
				sigma_init_aa=1e-3,
				pop_size=25,
				damp=0.1,
				damp_limit=0.05,
				damp_aa=0.2,
				damp_limit_aa=0.1,
				parents=5,
				):
		self.sigma_init = sigma_init
		self.pop_size = pop_size
		self.damp = damp
		self.damp_limit = damp_limit
		self.parents = parents
		self.action_dim = action_dim
		self.max_action = max_action
		self.sigma_init_aa = sigma_init_aa
		self.damp_aa = damp_aa
		self.damp_limit_aa = damp_limit_aa
		self.max_action_aa = max_action_aa

	def search(self, batch_size, mu_init, critic, visualize_func=None, n_iter=2, action_bound=True, visualize=False, elitism=False):
		cem = sepCEM(self.action_dim, batch_size, mu_init, self.sigma_init, self.sigma_init_aa, pop_size=self.pop_size, damp=self.damp, damp_limit=self.damp_limit, parents=self.parents, elitism=elitism,
			damp_limit_aa=self.damp_limit_aa, damp_aa = self.damp_aa)
		img_arr = []


		elite_actions = np.zeros((batch_size, n_iter, self.action_dim))
		elite_actions_scores = np.zeros((batch_size, n_iter))
		for ii in range(n_iter):
			actions = cem.ask(self.pop_size)
			if action_bound:
				actions[:, :, :3] = np.clip(actions[:, :, :3], -self.max_action, self.max_action)
				actions[:, :, 3:] = np.clip(actions[:, :, 3:], -self.max_action_aa, self.max_action_aa)
			Qs, pc_combined_best_all = critic(actions)
			best_action = cem.tell(actions, Qs)
			Qs_max = np.max(Qs, axis=-1)
			print(ii, np.mean(Qs_max))
			elite_actions[:, ii] = np.copy(cem.elite)
			elite_actions_scores[:, ii] = np.max(Qs, axis=-1)
			if visualize and visualize_func:
				img_arr.append(visualize_func(best_action[0]))
		cem_info_dict = {
			'cem_elite_pose': np.copy(elite_actions),
			'cem_elite_pose_scores': np.copy(elite_actions_scores),
			# 'pc_combined_best_all': pc_combined_best_all
		}
		if visualize:
			return img_arr, best_action, Qs_max, cem_info_dict
		else:
			return None, best_action, Qs_max, cem_info_dict
			# if iter == n_iter - 1:
				# best_Q = critic(best_action)
				# ori_Q = critic(action_init)

				# action_index = (best_Q < ori_Q).squeeze()
				# best_action[action_index] = action_init[action_index]
				# best_Q = torch.max(ori_Q, best_Q)

				# return best_action
