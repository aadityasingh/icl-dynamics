# Example cache

Consider a sequence classifier initialized as a 2L transformer with `d=64`, 8 heads per layer, and an MLP expansion factor of 4. On an input sequence of 3 exemplar-label pairs (where the last exemplar is the query), the sequence length would be `5` and the the output cache shapes for the highest level `SequenceClassifier` would look like:

```
{
	"embedding": [5, 64], 
	"example_embedding": [3, 64], 
	"examples": [3, 512], 
	"label_embedding": [3, 64], 
	"labels": [3], 
	"onehot_labels": [3, 5], 
	"out": [3, 5], 
	"pos_embedding": null, 
	"tok_embedding": [5, 64], 
	"transformer_output": {
		"block_outputs": [
			{ # Layer 1
				"attn_output": {
					"attn_post_drop": [1, 8, 5, 5], 
					"attn_pre_softmax": [1, 8, 5, 5], 
					"attn_scores": [1, 8, 5, 5], # _ x H x Query token x Key token
					"k": [1, 8, 5, 8], 
					"out": [5, 64], 
					"pre_rope_k": [1, 8, 5, 8], 
					"pre_rope_q": [1, 8, 5, 8], 
					"q": [1, 8, 5, 8], 
					"qkv": [3, 8, 5, 8],
					"v": [1, 8, 5, 8], 
					"values_post_proj": [5, 64], 
					"values_pre_proj": [5, 64], 
					"x": [5, 64]
				}, 
				"mlp": [5, 64], 
				"norm_inp": [5, 64], 
				"norm_residual_post_attn": [5, 64], 
				"out": [5, 64], 
				"residual_post_attn": [5, 64], 
				"x": [5, 64]
			}, 
			{ # Layer 2
				"attn_output": {
					"attn_post_drop": [1, 8, 5, 5], 
					"attn_pre_softmax": [1, 8, 5, 5], # _ x H x Query token x Key token
					"attn_scores": [1, 8, 5, 5], 
					"k": [1, 8, 5, 8], 
					"out": [5, 64], 
					"pre_rope_k": [1, 8, 5, 8], 
					"pre_rope_q": [1, 8, 5, 8], 
					"q": [1, 8, 5, 8], 
					"qkv": [3, 8, 5, 8], 
					"v": [1, 8, 5, 8], 
					"values_post_proj": [5, 64], 
					"values_pre_proj": [5, 64], 
					"x": [5, 64]
				}, 
				"mlp": [5, 64], # Currently intermediate MLP activations aren't cached, but could be easily added
				"norm_inp": [5, 64], 
				"norm_residual_post_attn": [5, 64], 
				"out": [5, 64], 
				"residual_post_attn": [5, 64], 
				"x": [5, 64]
			}
		], 
		"out": [5, 64], 
		"pre_unembed": [5, 64], 
		"x": [5, 64]
	}, 
	"unembedding": [5, 5]
}
```