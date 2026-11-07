
import torch
import torch.nn as nn

# SVD   : r(u,i) = mu + b_u + b_i + q_i . p_u
# SVD++ : r(u,i) = mu + b_u + b_i + q_i . (p_u + |N(u)|^-1/2 * sum_{j in N(u)} y_j)

class LatentFactorModel(nn.Module):

    # Parameters:
    # user_biases               b_u     (U+1, 1)
    # item_biases               b_i     (I+1, 1)
    # user_factors              p_u     (U+1, n_factors)
    # item_factors              q_i     (I+1, n_factors)
    # implicit_item_factors     y_j     (I+1, n_factors)        // for SVD++


    def __init__(self, n_user_rows, n_item_rows, n_factors, global_mean_rating, interacted_items_by_user=None,
                 initial_user_biases=None, initial_item_biases=None, initial_factor_std=0.01):

        super().__init__()
        self.global_mean_rating = global_mean_rating

        # set up parameter tables
        self.user_biases = nn.Embedding(n_user_rows, 1, padding_idx=0)
        self.item_biases = nn.Embedding(n_item_rows, 1, padding_idx=0)
        self.user_factors = nn.Embedding(n_user_rows, n_factors, padding_idx=0)
        self.item_factors = nn.Embedding(n_item_rows, n_factors, padding_idx=0)

        # biases start at 0, or at given values (e.g. the closed-form bias model) -> training starts from a good baseline
        nn.init.zeros_(self.user_biases.weight)
        nn.init.zeros_(self.item_biases.weight)
        with torch.no_grad():
            if initial_user_biases is not None:
                self.user_biases.weight[:, 0] = torch.as_tensor(initial_user_biases, dtype=torch.float32)
            if initial_item_biases is not None:
                self.item_biases.weight[:, 0] = torch.as_tensor(initial_item_biases, dtype=torch.float32)

        # factors start at small noise
        for factor_table in (self.user_factors, self.item_factors):
            nn.init.normal_(factor_table.weight, std=initial_factor_std)
            with torch.no_grad():
                factor_table.weight[0] = 0.0

        # SVD++ only: implicit item factors y_j & user's item list N(u)
        self.uses_implicit_feedback = interacted_items_by_user is not None
        if self.uses_implicit_feedback:
            self.implicit_item_factors = nn.EmbeddingBag(n_item_rows, n_factors, mode="sum")
            nn.init.normal_(self.implicit_item_factors.weight, std=initial_factor_std)
            self.interacted_items_by_user = [torch.as_tensor(items) for items in interacted_items_by_user]


    # For SVD++
    def implicit_feedback_vectors(self, user_rows):

        distinct_users, distinct_index_per_rating = torch.unique(user_rows, return_inverse=True)
        item_lists = [self.interacted_items_by_user[user] for user in distinct_users.tolist()]
        list_lengths = torch.tensor([len(items) for items in item_lists])

        list_starts = torch.cumsum(list_lengths, dim=0) - list_lengths
        normalisation = torch.repeat_interleave(1 / list_lengths.clamp(min=1).float().sqrt(), list_lengths)

        device = self.implicit_item_factors.weight.device

        implicit_vector_per_user = self.implicit_item_factors(torch.cat(item_lists).to(device), list_starts.to(device), per_sample_weights=normalisation.to(device))
        
        return implicit_vector_per_user[distinct_index_per_rating]


    def forward(self, user_rows, item_rows):

        user_vectors = self.user_factors(user_rows)
        
        if self.uses_implicit_feedback:
            user_vectors = user_vectors + self.implicit_feedback_vectors(user_rows)

        user_item_affinity = (user_vectors * self.item_factors(item_rows)).sum(dim=1)

        biases = self.user_biases(user_rows).squeeze(1) + self.item_biases(item_rows).squeeze(1)

        return self.global_mean_rating + biases + user_item_affinity


    def l2_penalty(self, user_rows, item_rows, user_penalty_weights, item_penalty_weights, bias_regularization, factor_regularization):

        # squared norms of the parameters used by each rating
        user_bias_squared = self.user_biases(user_rows).squeeze(1) ** 2
        item_bias_squared = self.item_biases(item_rows).squeeze(1) ** 2
        user_factor_squared = (self.user_factors(user_rows) ** 2).sum(dim=1)
        item_factor_squared = (self.item_factors(item_rows) ** 2).sum(dim=1)

        if self.uses_implicit_feedback:
            # y_j is regularised like q_j: through the ratings of item j
            item_factor_squared = item_factor_squared + (self.implicit_item_factors.weight[item_rows] ** 2).sum(dim=1)

        # weight per rating, then average over the batch
        bias_penalty = user_penalty_weights * user_bias_squared + item_penalty_weights * item_bias_squared
        factor_penalty = user_penalty_weights * user_factor_squared + item_penalty_weights * item_factor_squared
        
        return (bias_regularization * bias_penalty + factor_regularization * factor_penalty).mean()