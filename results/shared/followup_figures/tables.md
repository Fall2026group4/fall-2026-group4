# Follow-up results: tables generated from the saved CSVs

Every number below is read from a results file; nothing is typed by hand.

## Graded positive control (RepE, SAE-in-the-loop script)

| layer | lam | effect_E | control_q95 | p_upper | minimum_detectable_dose |
|---|---|---|---|---|---|
| 2.0 | 0.01 | 0.0008 | 0.0004 | 0.0067 | 0.01 |
| 2.0 | 0.02 | 0.0015 | 0.0008 | 0.0067 | 0.01 |
| 2.0 | 0.05 | 0.0036 | 0.002 | 0.0067 | 0.01 |
| 2.0 | 0.1 | 0.0069 | 0.0039 | 0.0067 | 0.01 |
| 2.0 | 0.25 | 0.0152 | 0.0087 | 0.0083 | 0.01 |
| 2.0 | 0.5 | 0.0254 | 0.0135 | 0.0083 | 0.01 |
| 2.0 | 1.0 | 0.0372 | 0.0124 | 0.0083 | 0.01 |
| 6.0 | 0.01 | 0.0045 | 0.0006 | 0.0017 | 0.01 |
| 6.0 | 0.02 | 0.0089 | 0.0012 | 0.0017 | 0.01 |
| 6.0 | 0.05 | 0.0219 | 0.003 | 0.0017 | 0.01 |
| 6.0 | 0.1 | 0.0426 | 0.0056 | 0.0017 | 0.01 |
| 6.0 | 0.25 | 0.0981 | 0.0113 | 0.0017 | 0.01 |
| 6.0 | 0.5 | 0.169 | 0.0129 | 0.0017 | 0.01 |
| 6.0 | 1.0 | 0.2336 | -0.0101 | 0.0017 | 0.01 |
| 10.0 | 0.01 | 0.0123 | 0.0008 | 0.0017 | 0.01 |
| 10.0 | 0.02 | 0.0245 | 0.0016 | 0.0017 | 0.01 |
| 10.0 | 0.05 | 0.0605 | 0.0038 | 0.0017 | 0.01 |
| 10.0 | 0.1 | 0.1182 | 0.0066 | 0.0017 | 0.01 |
| 10.0 | 0.25 | 0.2752 | 0.0101 | 0.0017 | 0.01 |
| 10.0 | 0.5 | 0.4826 | -0.0017 | 0.0017 | 0.01 |
| 10.0 | 1.0 | 0.698 | -0.0934 | 0.0017 | 0.01 |

## SAE-in-the-loop latent patching (RepE)

| layer | intervention | sae_architecture | sae_seed | control_group_kind | effect_E | share_of_activation_patch | p_upper | p_bh |
|---|---|---|---|---|---|---|---|---|
| 2 | latent_all | relu | 42.0 | gaussian_matched | -0.0001 | -0.0037 | 0.1581 | 0.1778 |
| 2 | error_term | relu | 42.0 | gaussian_matched | 0.0299 | 0.8049 | 0.0033 | 0.0053 |
| 2 | latent_top1 | relu | 42.0 | random_latents | -0.0077 | -0.207 | 1.0 | 1.0 |
| 2 | latent_top1 | relu | 42.0 | activation_matched_latents | -0.0077 | -0.207 | 0.9351 | 1.0 |
| 2 | latent_top10 | relu | 42.0 | random_latents | -0.039 | -1.0483 | 1.0 | 1.0 |
| 2 | latent_top10 | relu | 42.0 | activation_matched_latents | -0.039 | -1.0483 | 1.0 | 1.0 |
| 2 | latent_top50 | relu | 42.0 | random_latents | -0.0249 | -0.6713 | 1.0 | 1.0 |
| 2 | latent_top50 | relu | 42.0 | activation_matched_latents | -0.0249 | -0.6713 | 1.0 | 1.0 |
| 2 | latent_all | relu | 1042.0 | gaussian_matched | 0.0044 | 0.1174 | 0.1248 | 0.1484 |
| 2 | error_term | relu | 1042.0 | gaussian_matched | 0.0276 | 0.7425 | 0.0083 | 0.0125 |
| 2 | latent_top1 | relu | 1042.0 | random_latents | -0.0133 | -0.3583 | 1.0 | 1.0 |
| 2 | latent_top1 | relu | 1042.0 | activation_matched_latents | -0.0133 | -0.3583 | 0.9667 | 1.0 |
| 2 | latent_top10 | relu | 1042.0 | random_latents | -0.023 | -0.6191 | 1.0 | 1.0 |
| 2 | latent_top10 | relu | 1042.0 | activation_matched_latents | -0.023 | -0.6191 | 1.0 | 1.0 |
| 2 | latent_top50 | relu | 1042.0 | random_latents | -0.014 | -0.3758 | 1.0 | 1.0 |
| 2 | latent_top50 | relu | 1042.0 | activation_matched_latents | -0.014 | -0.3758 | 1.0 | 1.0 |
| 2 | latent_all | relu | 2042.0 | gaussian_matched | 0.0232 | 0.6257 | 0.0183 | 0.0235 |
| 2 | error_term | relu | 2042.0 | gaussian_matched | 0.0079 | 0.2115 | 0.1015 | 0.1142 |
| 2 | latent_top1 | relu | 2042.0 | random_latents | -0.015 | -0.4033 | 1.0 | 1.0 |
| 2 | latent_top1 | relu | 2042.0 | activation_matched_latents | -0.015 | -0.4033 | 0.9567 | 1.0 |
| 2 | latent_top10 | relu | 2042.0 | random_latents | -0.0311 | -0.8367 | 1.0 | 1.0 |
| 2 | latent_top10 | relu | 2042.0 | activation_matched_latents | -0.0311 | -0.8367 | 1.0 | 1.0 |
| 2 | latent_top50 | relu | 2042.0 | random_latents | -0.0064 | -0.1724 | 1.0 | 1.0 |
| 2 | latent_top50 | relu | 2042.0 | activation_matched_latents | -0.0064 | -0.1724 | 1.0 | 1.0 |
| 2 | latent_all | topk | 42.0 | gaussian_matched | -0.0295 | -0.7929 | 0.3161 | 0.3283 |
| 2 | error_term | topk | 42.0 | gaussian_matched | 0.0118 | 0.318 | 0.0383 | 0.047 |
| 2 | latent_top1 | topk | 42.0 | random_latents | -0.0191 | -0.5144 | 1.0 | 1.0 |
| 2 | latent_top1 | topk | 42.0 | activation_matched_latents | -0.0191 | -0.5144 | 1.0 | 1.0 |
| 2 | latent_top10 | topk | 42.0 | random_latents | -0.0376 | -1.0117 | 1.0 | 1.0 |
| 2 | latent_top10 | topk | 42.0 | activation_matched_latents | -0.0376 | -1.0117 | 1.0 | 1.0 |
| 2 | latent_top50 | topk | 42.0 | random_latents | -0.0317 | -0.8536 | 1.0 | 1.0 |
| 2 | latent_top50 | topk | 42.0 | activation_matched_latents | -0.0317 | -0.8536 | 1.0 | 1.0 |
| 2 | latent_all | topk | 1042.0 | gaussian_matched | -0.0277 | -0.7451 | 0.3378 | 0.3378 |
| 2 | error_term | topk | 1042.0 | gaussian_matched | 0.0089 | 0.2388 | 0.0815 | 0.0957 |
| 2 | latent_top1 | topk | 1042.0 | random_latents | -0.0209 | -0.5627 | 1.0 | 1.0 |
| 2 | latent_top1 | topk | 1042.0 | activation_matched_latents | -0.0209 | -0.5627 | 0.9734 | 1.0 |
| 2 | latent_top10 | topk | 1042.0 | random_latents | -0.0373 | -1.0029 | 1.0 | 1.0 |
| 2 | latent_top10 | topk | 1042.0 | activation_matched_latents | -0.0373 | -1.0029 | 1.0 | 1.0 |
| 2 | latent_top50 | topk | 1042.0 | random_latents | -0.0183 | -0.4932 | 1.0 | 1.0 |
| 2 | latent_top50 | topk | 1042.0 | activation_matched_latents | -0.0183 | -0.4932 | 1.0 | 1.0 |
| 2 | latent_all | topk | 2042.0 | gaussian_matched | 0.0234 | 0.6307 | 0.0133 | 0.0189 |
| 2 | error_term | topk | 2042.0 | gaussian_matched | -0.0338 | -0.9092 | 0.5857 | 0.5857 |
| 2 | latent_top1 | topk | 2042.0 | random_latents | -0.0224 | -0.6029 | 1.0 | 1.0 |
| 2 | latent_top1 | topk | 2042.0 | activation_matched_latents | -0.0224 | -0.6029 | 1.0 | 1.0 |
| 2 | latent_top10 | topk | 2042.0 | random_latents | -0.0428 | -1.1516 | 1.0 | 1.0 |
| 2 | latent_top10 | topk | 2042.0 | activation_matched_latents | -0.0428 | -1.1516 | 1.0 | 1.0 |
| 2 | latent_top50 | topk | 2042.0 | random_latents | -0.0021 | -0.056 | 0.8369 | 1.0 |
| 2 | latent_top50 | topk | 2042.0 | activation_matched_latents | -0.0021 | -0.056 | 0.8519 | 1.0 |
| 2 | latent_all | jumprelu | 42.0 | gaussian_matched | -0.0058 | -0.157 | 0.1265 | 0.1484 |
| 2 | error_term | jumprelu | 42.0 | gaussian_matched | 0.0288 | 0.7741 | 0.0166 | 0.0214 |
| 2 | latent_top1 | jumprelu | 42.0 | random_latents | -0.0423 | -1.1373 | 1.0 | 1.0 |
| 2 | latent_top1 | jumprelu | 42.0 | activation_matched_latents | -0.0423 | -1.1373 | 1.0 | 1.0 |
| 2 | latent_top10 | jumprelu | 42.0 | random_latents | -0.0496 | -1.3359 | 1.0 | 1.0 |
| 2 | latent_top10 | jumprelu | 42.0 | activation_matched_latents | -0.0496 | -1.3359 | 1.0 | 1.0 |
| 2 | latent_top50 | jumprelu | 42.0 | random_latents | -0.0464 | -1.2494 | 1.0 | 1.0 |
| 2 | latent_top50 | jumprelu | 42.0 | activation_matched_latents | -0.0464 | -1.2494 | 1.0 | 1.0 |
| 2 | latent_all | jumprelu | 1042.0 | gaussian_matched | -0.0091 | -0.244 | 0.178 | 0.1923 |
| 2 | error_term | jumprelu | 1042.0 | gaussian_matched | 0.0013 | 0.0359 | 0.1597 | 0.1725 |
| 2 | latent_top1 | jumprelu | 1042.0 | random_latents | -0.0332 | -0.8924 | 1.0 | 1.0 |
| 2 | latent_top1 | jumprelu | 1042.0 | activation_matched_latents | -0.0332 | -0.8924 | 1.0 | 1.0 |
| 2 | latent_top10 | jumprelu | 1042.0 | random_latents | -0.0312 | -0.8389 | 1.0 | 1.0 |
| 2 | latent_top10 | jumprelu | 1042.0 | activation_matched_latents | -0.0312 | -0.8389 | 1.0 | 1.0 |
| 2 | latent_top50 | jumprelu | 1042.0 | random_latents | -0.0336 | -0.9031 | 1.0 | 1.0 |
| 2 | latent_top50 | jumprelu | 1042.0 | activation_matched_latents | -0.0336 | -0.9031 | 1.0 | 1.0 |
| 2 | latent_all | jumprelu | 2042.0 | gaussian_matched | 0.0253 | 0.6806 | 0.015 | 0.0202 |
| 2 | error_term | jumprelu | 2042.0 | gaussian_matched | -0.0122 | -0.3278 | 0.3228 | 0.3352 |
| 2 | latent_top1 | jumprelu | 2042.0 | random_latents | -0.0039 | -0.1038 | 0.9884 | 1.0 |
| 2 | latent_top1 | jumprelu | 2042.0 | activation_matched_latents | -0.0039 | -0.1038 | 1.0 | 1.0 |
| 2 | latent_top10 | jumprelu | 2042.0 | random_latents | -0.0249 | -0.6697 | 1.0 | 1.0 |
| 2 | latent_top10 | jumprelu | 2042.0 | activation_matched_latents | -0.0249 | -0.6697 | 1.0 | 1.0 |
| 2 | latent_top50 | jumprelu | 2042.0 | random_latents | -0.0285 | -0.7666 | 1.0 | 1.0 |
| 2 | latent_top50 | jumprelu | 2042.0 | activation_matched_latents | -0.0285 | -0.7666 | 1.0 | 1.0 |
| 6 | latent_all | relu | 42.0 | gaussian_matched | 0.1102 | 0.4716 | 0.0017 | 0.0026 |
| 6 | error_term | relu | 42.0 | gaussian_matched | 0.1171 | 0.5014 | 0.0017 | 0.0028 |
| 6 | latent_top1 | relu | 42.0 | random_latents | -0.0163 | -0.0698 | 1.0 | 1.0 |
| 6 | latent_top1 | relu | 42.0 | activation_matched_latents | -0.0163 | -0.0698 | 1.0 | 1.0 |
| 6 | latent_top10 | relu | 42.0 | random_latents | -0.0119 | -0.051 | 1.0 | 1.0 |
| 6 | latent_top10 | relu | 42.0 | activation_matched_latents | -0.0119 | -0.051 | 1.0 | 1.0 |
| 6 | latent_top50 | relu | 42.0 | random_latents | -0.0132 | -0.0564 | 1.0 | 1.0 |
| 6 | latent_top50 | relu | 42.0 | activation_matched_latents | -0.0132 | -0.0564 | 1.0 | 1.0 |
| 6 | latent_all | relu | 1042.0 | gaussian_matched | 0.0902 | 0.3863 | 0.0017 | 0.0026 |
| 6 | error_term | relu | 1042.0 | gaussian_matched | 0.1418 | 0.6072 | 0.0017 | 0.0028 |
| 6 | latent_top1 | relu | 1042.0 | random_latents | -0.0077 | -0.0331 | 1.0 | 1.0 |
| 6 | latent_top1 | relu | 1042.0 | activation_matched_latents | -0.0077 | -0.0331 | 1.0 | 1.0 |
| 6 | latent_top10 | relu | 1042.0 | random_latents | -0.0138 | -0.0591 | 1.0 | 1.0 |
| 6 | latent_top10 | relu | 1042.0 | activation_matched_latents | -0.0138 | -0.0591 | 1.0 | 1.0 |
| 6 | latent_top50 | relu | 1042.0 | random_latents | -0.0158 | -0.0675 | 1.0 | 1.0 |
| 6 | latent_top50 | relu | 1042.0 | activation_matched_latents | -0.0158 | -0.0675 | 1.0 | 1.0 |
| 6 | latent_all | relu | 2042.0 | gaussian_matched | 0.0947 | 0.4054 | 0.0017 | 0.0026 |
| 6 | error_term | relu | 2042.0 | gaussian_matched | 0.1459 | 0.6244 | 0.0017 | 0.0028 |
| 6 | latent_top1 | relu | 2042.0 | random_latents | -0.0166 | -0.0709 | 1.0 | 1.0 |
| 6 | latent_top1 | relu | 2042.0 | activation_matched_latents | -0.0166 | -0.0709 | 1.0 | 1.0 |
| 6 | latent_top10 | relu | 2042.0 | random_latents | -0.0338 | -0.1447 | 1.0 | 1.0 |
| 6 | latent_top10 | relu | 2042.0 | activation_matched_latents | -0.0338 | -0.1447 | 1.0 | 1.0 |
| 6 | latent_top50 | relu | 2042.0 | random_latents | -0.0124 | -0.0531 | 1.0 | 1.0 |
| 6 | latent_top50 | relu | 2042.0 | activation_matched_latents | -0.0124 | -0.0531 | 1.0 | 1.0 |
| 6 | latent_all | topk | 42.0 | gaussian_matched | -0.0229 | -0.098 | 0.0033 | 0.005 |
| 6 | error_term | topk | 42.0 | gaussian_matched | 0.0517 | 0.2211 | 0.0017 | 0.0028 |
| 6 | latent_top1 | topk | 42.0 | random_latents | -0.0295 | -0.1263 | 1.0 | 1.0 |
| 6 | latent_top1 | topk | 42.0 | activation_matched_latents | -0.0295 | -0.1263 | 1.0 | 1.0 |
| 6 | latent_top10 | topk | 42.0 | random_latents | -0.0442 | -0.1893 | 1.0 | 1.0 |
| 6 | latent_top10 | topk | 42.0 | activation_matched_latents | -0.0442 | -0.1893 | 1.0 | 1.0 |
| 6 | latent_top50 | topk | 42.0 | random_latents | -0.0327 | -0.1401 | 1.0 | 1.0 |
| 6 | latent_top50 | topk | 42.0 | activation_matched_latents | -0.0327 | -0.1401 | 1.0 | 1.0 |
| 6 | latent_all | topk | 1042.0 | gaussian_matched | 0.0481 | 0.2059 | 0.0017 | 0.0026 |
| 6 | error_term | topk | 1042.0 | gaussian_matched | 0.0376 | 0.1609 | 0.0017 | 0.0028 |
| 6 | latent_top1 | topk | 1042.0 | random_latents | -0.0035 | -0.015 | 0.9967 | 1.0 |
| 6 | latent_top1 | topk | 1042.0 | activation_matched_latents | -0.0035 | -0.015 | 0.8336 | 1.0 |
| 6 | latent_top10 | topk | 1042.0 | random_latents | -0.0475 | -0.2033 | 1.0 | 1.0 |
| 6 | latent_top10 | topk | 1042.0 | activation_matched_latents | -0.0475 | -0.2033 | 1.0 | 1.0 |
| 6 | latent_top50 | topk | 1042.0 | random_latents | -0.0298 | -0.1276 | 1.0 | 1.0 |
| 6 | latent_top50 | topk | 1042.0 | activation_matched_latents | -0.0298 | -0.1276 | 1.0 | 1.0 |
| 6 | latent_all | topk | 2042.0 | gaussian_matched | 0.0219 | 0.0938 | 0.0017 | 0.0026 |
| 6 | error_term | topk | 2042.0 | gaussian_matched | -0.0262 | -0.1119 | 0.015 | 0.0202 |
| 6 | latent_top1 | topk | 2042.0 | random_latents | -0.073 | -0.3123 | 1.0 | 1.0 |
| 6 | latent_top1 | topk | 2042.0 | activation_matched_latents | -0.073 | -0.3123 | 1.0 | 1.0 |
| 6 | latent_top10 | topk | 2042.0 | random_latents | -0.0771 | -0.3299 | 1.0 | 1.0 |
| 6 | latent_top10 | topk | 2042.0 | activation_matched_latents | -0.0771 | -0.3299 | 1.0 | 1.0 |
| 6 | latent_top50 | topk | 2042.0 | random_latents | -0.073 | -0.3126 | 1.0 | 1.0 |
| 6 | latent_top50 | topk | 2042.0 | activation_matched_latents | -0.073 | -0.3126 | 1.0 | 1.0 |
| 6 | latent_all | jumprelu | 42.0 | gaussian_matched | 0.0792 | 0.339 | 0.0017 | 0.0026 |
| 6 | error_term | jumprelu | 42.0 | gaussian_matched | 0.0606 | 0.2594 | 0.0017 | 0.0028 |
| 6 | latent_top1 | jumprelu | 42.0 | random_latents | -0.0609 | -0.2607 | 1.0 | 1.0 |
| 6 | latent_top1 | jumprelu | 42.0 | activation_matched_latents | -0.0609 | -0.2607 | 1.0 | 1.0 |
| 6 | latent_top10 | jumprelu | 42.0 | random_latents | -0.0637 | -0.2726 | 1.0 | 1.0 |
| 6 | latent_top10 | jumprelu | 42.0 | activation_matched_latents | -0.0637 | -0.2726 | 1.0 | 1.0 |
| 6 | latent_top50 | jumprelu | 42.0 | random_latents | -0.0179 | -0.0766 | 1.0 | 1.0 |
| 6 | latent_top50 | jumprelu | 42.0 | activation_matched_latents | -0.0179 | -0.0766 | 1.0 | 1.0 |
| 6 | latent_all | jumprelu | 1042.0 | gaussian_matched | 0.1211 | 0.5184 | 0.0017 | 0.0026 |
| 6 | error_term | jumprelu | 1042.0 | gaussian_matched | 0.0157 | 0.0672 | 0.0017 | 0.0028 |
| 6 | latent_top1 | jumprelu | 1042.0 | random_latents | -0.0615 | -0.2634 | 1.0 | 1.0 |
| 6 | latent_top1 | jumprelu | 1042.0 | activation_matched_latents | -0.0615 | -0.2634 | 1.0 | 1.0 |
| 6 | latent_top10 | jumprelu | 1042.0 | random_latents | -0.0629 | -0.2694 | 1.0 | 1.0 |
| 6 | latent_top10 | jumprelu | 1042.0 | activation_matched_latents | -0.0629 | -0.2694 | 1.0 | 1.0 |
| 6 | latent_top50 | jumprelu | 1042.0 | random_latents | -0.0117 | -0.0501 | 1.0 | 1.0 |
| 6 | latent_top50 | jumprelu | 1042.0 | activation_matched_latents | -0.0117 | -0.0501 | 1.0 | 1.0 |
| 6 | latent_all | jumprelu | 2042.0 | gaussian_matched | 0.0863 | 0.3693 | 0.0017 | 0.0026 |
| 6 | error_term | jumprelu | 2042.0 | gaussian_matched | 0.0487 | 0.2085 | 0.0017 | 0.0028 |
| 6 | latent_top1 | jumprelu | 2042.0 | random_latents | -0.0461 | -0.1975 | 1.0 | 1.0 |
| 6 | latent_top1 | jumprelu | 2042.0 | activation_matched_latents | -0.0461 | -0.1975 | 1.0 | 1.0 |
| 6 | latent_top10 | jumprelu | 2042.0 | random_latents | -0.0508 | -0.2173 | 1.0 | 1.0 |
| 6 | latent_top10 | jumprelu | 2042.0 | activation_matched_latents | -0.0508 | -0.2173 | 1.0 | 1.0 |
| 6 | latent_top50 | jumprelu | 2042.0 | random_latents | -0.0259 | -0.1107 | 1.0 | 1.0 |
| 6 | latent_top50 | jumprelu | 2042.0 | activation_matched_latents | -0.0259 | -0.1107 | 1.0 | 1.0 |
| 10 | latent_all | relu | 42.0 | gaussian_matched | 0.4742 | 0.6794 | 0.0017 | 0.0026 |
| 10 | error_term | relu | 42.0 | gaussian_matched | 0.2557 | 0.3663 | 0.0017 | 0.0028 |
| 10 | latent_top1 | relu | 42.0 | random_latents | -0.0084 | -0.0121 | 1.0 | 1.0 |
| 10 | latent_top1 | relu | 42.0 | activation_matched_latents | -0.0084 | -0.0121 | 1.0 | 1.0 |
| 10 | latent_top10 | relu | 42.0 | random_latents | 0.0517 | 0.074 | 0.0017 | 0.009 |
| 10 | latent_top10 | relu | 42.0 | activation_matched_latents | 0.0517 | 0.074 | 0.0033 | 0.0245 |
| 10 | latent_top50 | relu | 42.0 | random_latents | 0.1317 | 0.1887 | 0.0017 | 0.009 |
| 10 | latent_top50 | relu | 42.0 | activation_matched_latents | 0.1317 | 0.1887 | 0.0017 | 0.0135 |
| 10 | latent_all | relu | 1042.0 | gaussian_matched | 0.468 | 0.6705 | 0.0017 | 0.0026 |
| 10 | error_term | relu | 1042.0 | gaussian_matched | 0.2539 | 0.3637 | 0.0017 | 0.0028 |
| 10 | latent_top1 | relu | 1042.0 | random_latents | -0.0077 | -0.011 | 1.0 | 1.0 |
| 10 | latent_top1 | relu | 1042.0 | activation_matched_latents | -0.0077 | -0.011 | 1.0 | 1.0 |
| 10 | latent_top10 | relu | 1042.0 | random_latents | 0.0582 | 0.0834 | 0.0017 | 0.009 |
| 10 | latent_top10 | relu | 1042.0 | activation_matched_latents | 0.0582 | 0.0834 | 0.0017 | 0.0135 |
| 10 | latent_top50 | relu | 1042.0 | random_latents | 0.0989 | 0.1417 | 0.0017 | 0.009 |
| 10 | latent_top50 | relu | 1042.0 | activation_matched_latents | 0.0989 | 0.1417 | 0.0017 | 0.0135 |
| 10 | latent_all | relu | 2042.0 | gaussian_matched | 0.4765 | 0.6826 | 0.0017 | 0.0026 |
| 10 | error_term | relu | 2042.0 | gaussian_matched | 0.2507 | 0.3591 | 0.0017 | 0.0028 |
| 10 | latent_top1 | relu | 2042.0 | random_latents | -0.0064 | -0.0092 | 1.0 | 1.0 |
| 10 | latent_top1 | relu | 2042.0 | activation_matched_latents | -0.0064 | -0.0092 | 1.0 | 1.0 |
| 10 | latent_top10 | relu | 2042.0 | random_latents | 0.0478 | 0.0685 | 0.0017 | 0.009 |
| 10 | latent_top10 | relu | 2042.0 | activation_matched_latents | 0.0478 | 0.0685 | 0.0017 | 0.0135 |
| 10 | latent_top50 | relu | 2042.0 | random_latents | 0.1035 | 0.1483 | 0.0017 | 0.009 |
| 10 | latent_top50 | relu | 2042.0 | activation_matched_latents | 0.1035 | 0.1483 | 0.0017 | 0.0135 |
| 10 | latent_all | topk | 42.0 | gaussian_matched | 0.3435 | 0.4921 | 0.0017 | 0.0026 |
| 10 | error_term | topk | 42.0 | gaussian_matched | -0.0637 | -0.0912 | 0.0017 | 0.0028 |
| 10 | latent_top1 | topk | 42.0 | random_latents | 0.0082 | 0.0118 | 0.005 | 0.0225 |
| 10 | latent_top1 | topk | 42.0 | activation_matched_latents | 0.0082 | 0.0118 | 0.2097 | 0.9989 |
| 10 | latent_top10 | topk | 42.0 | random_latents | 0.052 | 0.0744 | 0.0017 | 0.009 |
| 10 | latent_top10 | topk | 42.0 | activation_matched_latents | 0.052 | 0.0744 | 0.1913 | 0.9989 |
| 10 | latent_top50 | topk | 42.0 | random_latents | 0.1556 | 0.223 | 0.0017 | 0.009 |
| 10 | latent_top50 | topk | 42.0 | activation_matched_latents | 0.1556 | 0.223 | 0.0017 | 0.0135 |
| 10 | latent_all | topk | 1042.0 | gaussian_matched | 0.2768 | 0.3965 | 0.0017 | 0.0026 |
| 10 | error_term | topk | 1042.0 | gaussian_matched | 0.0322 | 0.0462 | 0.0017 | 0.0028 |
| 10 | latent_top1 | topk | 1042.0 | random_latents | 0.0006 | 0.0008 | 0.1065 | 0.3921 |
| 10 | latent_top1 | topk | 1042.0 | activation_matched_latents | 0.0006 | 0.0008 | 0.8136 | 1.0 |
| 10 | latent_top10 | topk | 1042.0 | random_latents | 0.0292 | 0.0419 | 0.0067 | 0.0284 |
| 10 | latent_top10 | topk | 1042.0 | activation_matched_latents | 0.0292 | 0.0419 | 0.594 | 1.0 |
| 10 | latent_top50 | topk | 1042.0 | random_latents | 0.1527 | 0.2188 | 0.0017 | 0.009 |
| 10 | latent_top50 | topk | 1042.0 | activation_matched_latents | 0.1527 | 0.2188 | 0.0017 | 0.0135 |
| 10 | latent_all | topk | 2042.0 | gaussian_matched | 0.3472 | 0.4974 | 0.0017 | 0.0026 |
| 10 | error_term | topk | 2042.0 | gaussian_matched | -0.1074 | -0.1539 | 0.0116 | 0.0166 |
| 10 | latent_top1 | topk | 2042.0 | random_latents | 0.0178 | 0.0255 | 0.0033 | 0.0159 |
| 10 | latent_top1 | topk | 2042.0 | activation_matched_latents | 0.0178 | 0.0255 | 0.1664 | 0.9627 |
| 10 | latent_top10 | topk | 2042.0 | random_latents | 0.06 | 0.086 | 0.0017 | 0.009 |
| 10 | latent_top10 | topk | 2042.0 | activation_matched_latents | 0.06 | 0.086 | 0.0849 | 0.5728 |
| 10 | latent_top50 | topk | 2042.0 | random_latents | 0.158 | 0.2263 | 0.0017 | 0.009 |
| 10 | latent_top50 | topk | 2042.0 | activation_matched_latents | 0.158 | 0.2263 | 0.0017 | 0.0135 |
| 10 | latent_all | jumprelu | 42.0 | gaussian_matched | 0.3221 | 0.4614 | 0.0017 | 0.0026 |
| 10 | error_term | jumprelu | 42.0 | gaussian_matched | 0.0207 | 0.0297 | 0.0017 | 0.0028 |
| 10 | latent_top1 | jumprelu | 42.0 | random_latents | 0.007 | 0.0101 | 0.0133 | 0.0539 |
| 10 | latent_top1 | jumprelu | 42.0 | activation_matched_latents | 0.007 | 0.0101 | 0.203 | 0.9989 |
| 10 | latent_top10 | jumprelu | 42.0 | random_latents | 0.022 | 0.0315 | 0.0166 | 0.0642 |
| 10 | latent_top10 | jumprelu | 42.0 | activation_matched_latents | 0.022 | 0.0315 | 0.8552 | 1.0 |
| 10 | latent_top50 | jumprelu | 42.0 | random_latents | 0.1424 | 0.204 | 0.0017 | 0.009 |
| 10 | latent_top50 | jumprelu | 42.0 | activation_matched_latents | 0.1424 | 0.204 | 0.0017 | 0.0135 |
| 10 | latent_all | jumprelu | 1042.0 | gaussian_matched | 0.3621 | 0.5188 | 0.0017 | 0.0026 |
| 10 | error_term | jumprelu | 1042.0 | gaussian_matched | 0.0336 | 0.0482 | 0.0017 | 0.0028 |
| 10 | latent_top1 | jumprelu | 1042.0 | random_latents | -0.0008 | -0.0011 | 0.9418 | 1.0 |
| 10 | latent_top1 | jumprelu | 1042.0 | activation_matched_latents | -0.0008 | -0.0011 | 0.8785 | 1.0 |
| 10 | latent_top10 | jumprelu | 1042.0 | random_latents | 0.0324 | 0.0464 | 0.0033 | 0.0159 |
| 10 | latent_top10 | jumprelu | 1042.0 | activation_matched_latents | 0.0324 | 0.0464 | 0.5075 | 1.0 |
| 10 | latent_top50 | jumprelu | 1042.0 | random_latents | 0.1099 | 0.1575 | 0.0017 | 0.009 |
| 10 | latent_top50 | jumprelu | 1042.0 | activation_matched_latents | 0.1099 | 0.1575 | 0.1381 | 0.8605 |
| 10 | latent_all | jumprelu | 2042.0 | gaussian_matched | 0.379 | 0.5429 | 0.0017 | 0.0026 |
| 10 | error_term | jumprelu | 2042.0 | gaussian_matched | 0.0641 | 0.0919 | 0.0017 | 0.0028 |
| 10 | latent_top1 | jumprelu | 2042.0 | random_latents | -0.006 | -0.0086 | 1.0 | 1.0 |
| 10 | latent_top1 | jumprelu | 2042.0 | activation_matched_latents | -0.006 | -0.0086 | 0.9767 | 1.0 |
| 10 | latent_top10 | jumprelu | 2042.0 | random_latents | 0.0419 | 0.06 | 0.0017 | 0.009 |
| 10 | latent_top10 | jumprelu | 2042.0 | activation_matched_latents | 0.0419 | 0.06 | 0.3544 | 1.0 |
| 10 | latent_top50 | jumprelu | 2042.0 | random_latents | 0.1392 | 0.1995 | 0.0017 | 0.009 |
| 10 | latent_top50 | jumprelu | 2042.0 | activation_matched_latents | 0.1392 | 0.1995 | 0.0017 | 0.0135 |

## Direction steering, both signs (RepE)

| layer | intervention | sae_architecture | sae_seed | sign | effect_E | p_two_sided | p_bh |
|---|---|---|---|---|---|---|---|
| 2 | raw_honesty_direction |  |  | 1 | -0.023 | 0.7587 | 0.6223 |
| 2 | raw_honesty_direction |  |  | -1 | -0.1055 | 0.0033 |  |
| 2 | sae_reconstruction_direction | relu | 42.0 | 1 | -0.0372 | 0.3794 | 0.5208 |
| 2 | sae_reconstruction_direction | relu | 42.0 | -1 | -0.1826 | 0.0033 |  |
| 2 | sae_top1_direction | relu | 42.0 | 1 | -0.0214 | 0.8486 | 0.8812 |
| 2 | sae_top1_direction | relu | 42.0 | -1 | -0.0398 | 0.3028 |  |
| 2 | sae_top10_direction | relu | 42.0 | 1 | -0.041 | 0.173 | 0.3504 |
| 2 | sae_top10_direction | relu | 42.0 | -1 | -0.1386 | 0.0033 |  |
| 2 | sae_reconstruction_direction | relu | 1042.0 | 1 | -0.039 | 0.3261 | 0.4803 |
| 2 | sae_reconstruction_direction | relu | 1042.0 | -1 | -0.1778 | 0.0033 |  |
| 2 | sae_top1_direction | relu | 1042.0 | 1 | -0.0249 | 0.7521 | 0.8122 |
| 2 | sae_top1_direction | relu | 1042.0 | -1 | -0.0383 | 0.3727 |  |
| 2 | sae_top10_direction | relu | 1042.0 | 1 | -0.0382 | 0.2596 | 0.438 |
| 2 | sae_top10_direction | relu | 1042.0 | -1 | -0.1475 | 0.0033 |  |
| 2 | sae_reconstruction_direction | relu | 2042.0 | 1 | -0.0367 | 0.3927 | 0.5214 |
| 2 | sae_reconstruction_direction | relu | 2042.0 | -1 | -0.1662 | 0.0033 |  |
| 2 | sae_top1_direction | relu | 2042.0 | 1 | -0.0263 | 0.6988 | 0.7649 |
| 2 | sae_top1_direction | relu | 2042.0 | -1 | -0.0338 | 0.4825 |  |
| 2 | sae_top10_direction | relu | 2042.0 | 1 | -0.0406 | 0.2097 | 0.3949 |
| 2 | sae_top10_direction | relu | 2042.0 | -1 | -0.1256 | 0.0033 |  |
| 2 | sae_reconstruction_direction | topk | 42.0 | 1 | -0.038 | 0.3561 | 0.506 |
| 2 | sae_reconstruction_direction | topk | 42.0 | -1 | -0.1813 | 0.0033 |  |
| 2 | sae_top1_direction | topk | 42.0 | 1 | -0.0712 | 0.0266 | 0.098 |
| 2 | sae_top1_direction | topk | 42.0 | -1 | -0.0118 | 0.7488 |  |
| 2 | sae_top10_direction | topk | 42.0 | 1 | -0.0521 | 0.0998 | 0.2378 |
| 2 | sae_top10_direction | topk | 42.0 | -1 | -0.1388 | 0.0033 |  |
| 2 | sae_reconstruction_direction | topk | 1042.0 | 1 | -0.0335 | 0.4859 | 0.5963 |
| 2 | sae_reconstruction_direction | topk | 1042.0 | -1 | -0.1402 | 0.0033 |  |
| 2 | sae_top1_direction | topk | 1042.0 | 1 | -0.0329 | 0.4792 | 0.5963 |
| 2 | sae_top1_direction | topk | 1042.0 | -1 | -0.0261 | 0.772 |  |
| 2 | sae_top10_direction | topk | 1042.0 | 1 | -0.0536 | 0.0832 | 0.2106 |
| 2 | sae_top10_direction | topk | 1042.0 | -1 | -0.1101 | 0.0033 |  |
| 2 | sae_reconstruction_direction | topk | 2042.0 | 1 | -0.0396 | 0.3161 | 0.4742 |
| 2 | sae_reconstruction_direction | topk | 2042.0 | -1 | -0.1895 | 0.0033 |  |
| 2 | sae_top1_direction | topk | 2042.0 | 1 | -0.0398 | 0.2795 | 0.4528 |
| 2 | sae_top1_direction | topk | 2042.0 | -1 | -0.0873 | 0.01 |  |
| 2 | sae_top10_direction | topk | 2042.0 | 1 | -0.0558 | 0.0599 | 0.1797 |
| 2 | sae_top10_direction | topk | 2042.0 | -1 | -0.1833 | 0.0033 |  |
| 2 | sae_reconstruction_direction | jumprelu | 42.0 | 1 | -0.0435 | 0.2363 | 0.435 |
| 2 | sae_reconstruction_direction | jumprelu | 42.0 | -1 | -0.2438 | 0.0033 |  |
| 2 | sae_top1_direction | jumprelu | 42.0 | 1 | -0.0538 | 0.1065 | 0.2464 |
| 2 | sae_top1_direction | jumprelu | 42.0 | -1 | -0.1954 | 0.01 |  |
| 2 | sae_top10_direction | jumprelu | 42.0 | 1 | -0.0572 | 0.0765 | 0.2067 |
| 2 | sae_top10_direction | jumprelu | 42.0 | -1 | -0.3104 | 0.0033 |  |
| 2 | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | -0.0328 | 0.5025 | 0.6075 |
| 2 | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | -0.2377 | 0.0033 |  |
| 2 | sae_top1_direction | jumprelu | 1042.0 | 1 | -0.0494 | 0.0799 | 0.2087 |
| 2 | sae_top1_direction | jumprelu | 1042.0 | -1 | -0.1267 | 0.0033 |  |
| 2 | sae_top10_direction | jumprelu | 1042.0 | 1 | -0.0508 | 0.1631 | 0.3387 |
| 2 | sae_top10_direction | jumprelu | 1042.0 | -1 | -0.2452 | 0.0033 |  |
| 2 | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | -0.0373 | 0.3794 | 0.5208 |
| 2 | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | -0.2311 | 0.0033 |  |
| 2 | sae_top1_direction | jumprelu | 2042.0 | 1 | -0.0181 | 0.8985 | 0.9097 |
| 2 | sae_top1_direction | jumprelu | 2042.0 | -1 | -0.0179 | 0.8985 |  |
| 2 | sae_top10_direction | jumprelu | 2042.0 | 1 | -0.0354 | 0.3128 | 0.4742 |
| 2 | sae_top10_direction | jumprelu | 2042.0 | -1 | -0.1654 | 0.0033 |  |
| 6 | raw_honesty_direction |  |  | 1 | 0.0048 | 0.01 | 0.0075 |
| 6 | raw_honesty_direction |  |  | -1 | -0.0067 | 0.0033 |  |
| 6 | sae_reconstruction_direction | relu | 42.0 | 1 | -0.0017 | 0.4659 | 0.5896 |
| 6 | sae_reconstruction_direction | relu | 42.0 | -1 | -0.0021 | 0.3527 |  |
| 6 | sae_top1_direction | relu | 42.0 | 1 | -0.0026 | 0.3128 | 0.4742 |
| 6 | sae_top1_direction | relu | 42.0 | -1 | -0.0 | 0.8952 |  |
| 6 | sae_top10_direction | relu | 42.0 | 1 | 0.0007 | 0.7953 | 0.8367 |
| 6 | sae_top10_direction | relu | 42.0 | -1 | -0.0041 | 0.0765 |  |
| 6 | sae_reconstruction_direction | relu | 1042.0 | 1 | -0.0015 | 0.5191 | 0.6094 |
| 6 | sae_reconstruction_direction | relu | 1042.0 | -1 | -0.0023 | 0.3095 |  |
| 6 | sae_top1_direction | relu | 1042.0 | 1 | -0.0021 | 0.4426 | 0.5782 |
| 6 | sae_top1_direction | relu | 1042.0 | -1 | -0.0003 | 0.9983 |  |
| 6 | sae_top10_direction | relu | 1042.0 | 1 | -0.0009 | 0.6922 | 0.7649 |
| 6 | sae_top10_direction | relu | 1042.0 | -1 | -0.0024 | 0.2596 |  |
| 6 | sae_reconstruction_direction | relu | 2042.0 | 1 | -0.0024 | 0.3028 | 0.4742 |
| 6 | sae_reconstruction_direction | relu | 2042.0 | -1 | -0.0014 | 0.5657 |  |
| 6 | sae_top1_direction | relu | 2042.0 | 1 | -0.0027 | 0.2696 | 0.4456 |
| 6 | sae_top1_direction | relu | 2042.0 | -1 | 0.0001 | 0.8852 |  |
| 6 | sae_top10_direction | relu | 2042.0 | 1 | -0.002 | 0.2463 | 0.438 |
| 6 | sae_top10_direction | relu | 2042.0 | -1 | -0.0015 | 0.3594 |  |
| 6 | sae_reconstruction_direction | topk | 42.0 | 1 | 0.0023 | 0.2097 | 0.3949 |
| 6 | sae_reconstruction_direction | topk | 42.0 | -1 | -0.0046 | 0.0333 |  |
| 6 | sae_top1_direction | topk | 42.0 | 1 | -0.004 | 0.0466 | 0.1572 |
| 6 | sae_top1_direction | topk | 42.0 | -1 | 0.0013 | 0.3661 |  |
| 6 | sae_top10_direction | topk | 42.0 | 1 | -0.0016 | 0.3927 | 0.5214 |
| 6 | sae_top10_direction | topk | 42.0 | -1 | -0.0014 | 0.4759 |  |
| 6 | sae_reconstruction_direction | topk | 1042.0 | 1 | 0.0009 | 0.6256 | 0.7038 |
| 6 | sae_reconstruction_direction | topk | 1042.0 | -1 | -0.0036 | 0.1265 |  |
| 6 | sae_top1_direction | topk | 1042.0 | 1 | -0.0016 | 0.4659 | 0.5896 |
| 6 | sae_top1_direction | topk | 1042.0 | -1 | 0.0005 | 0.7055 |  |
| 6 | sae_top10_direction | topk | 1042.0 | 1 | -0.0058 | 0.0067 | 0.0449 |
| 6 | sae_top10_direction | topk | 1042.0 | -1 | 0.0027 | 0.1364 |  |
| 6 | sae_reconstruction_direction | topk | 2042.0 | 1 | 0.0005 | 0.7754 | 0.8264 |
| 6 | sae_reconstruction_direction | topk | 2042.0 | -1 | -0.0034 | 0.1597 |  |
| 6 | sae_top1_direction | topk | 2042.0 | 1 | -0.0064 | 0.0067 | 0.0449 |
| 6 | sae_top1_direction | topk | 2042.0 | -1 | 0.0024 | 0.1797 |  |
| 6 | sae_top10_direction | topk | 2042.0 | 1 | -0.0055 | 0.0233 | 0.098 |
| 6 | sae_top10_direction | topk | 2042.0 | -1 | 0.0016 | 0.3361 |  |
| 6 | sae_reconstruction_direction | jumprelu | 42.0 | 1 | 0.0029 | 0.1265 | 0.2768 |
| 6 | sae_reconstruction_direction | jumprelu | 42.0 | -1 | -0.0053 | 0.0166 |  |
| 6 | sae_top1_direction | jumprelu | 42.0 | 1 | -0.0057 | 0.0033 | 0.03 |
| 6 | sae_top1_direction | jumprelu | 42.0 | -1 | 0.0024 | 0.193 |  |
| 6 | sae_top10_direction | jumprelu | 42.0 | 1 | -0.0031 | 0.1464 | 0.3121 |
| 6 | sae_top10_direction | jumprelu | 42.0 | -1 | -0.0001 | 0.9318 |  |
| 6 | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | 0.0012 | 0.5391 | 0.6238 |
| 6 | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | -0.0037 | 0.1198 |  |
| 6 | sae_top1_direction | jumprelu | 1042.0 | 1 | -0.0064 | 0.0133 | 0.077 |
| 6 | sae_top1_direction | jumprelu | 1042.0 | -1 | 0.0028 | 0.1464 |  |
| 6 | sae_top10_direction | jumprelu | 1042.0 | 1 | -0.0024 | 0.2562 | 0.438 |
| 6 | sae_top10_direction | jumprelu | 1042.0 | -1 | -0.0008 | 0.6789 |  |
| 6 | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | 0.003 | 0.1198 | 0.2696 |
| 6 | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | -0.0054 | 0.0166 |  |
| 6 | sae_top1_direction | jumprelu | 2042.0 | 1 | -0.0051 | 0.02 | 0.0951 |
| 6 | sae_top1_direction | jumprelu | 2042.0 | -1 | 0.0024 | 0.1597 |  |
| 6 | sae_top10_direction | jumprelu | 2042.0 | 1 | 0.0 | 0.9218 | 0.9218 |
| 6 | sae_top10_direction | jumprelu | 2042.0 | -1 | -0.0032 | 0.1032 |  |
| 10 | raw_honesty_direction |  |  | 1 | 0.1072 | 0.0033 | 0.005 |
| 10 | raw_honesty_direction |  |  | -1 | -0.5997 | 0.0033 |  |
| 10 | sae_reconstruction_direction | relu | 42.0 | 1 | -0.0362 | 0.0067 | 0.0449 |
| 10 | sae_reconstruction_direction | relu | 42.0 | -1 | -0.5582 | 0.0033 |  |
| 10 | sae_top1_direction | relu | 42.0 | 1 | -0.4381 | 0.0266 | 0.098 |
| 10 | sae_top1_direction | relu | 42.0 | -1 | -0.3329 | 0.0865 |  |
| 10 | sae_top10_direction | relu | 42.0 | 1 | -0.3808 | 0.0599 | 0.1797 |
| 10 | sae_top10_direction | relu | 42.0 | -1 | -0.4442 | 0.03 |  |
| 10 | sae_reconstruction_direction | relu | 1042.0 | 1 | -0.0677 | 0.03 | 0.1055 |
| 10 | sae_reconstruction_direction | relu | 1042.0 | -1 | -0.5345 | 0.0033 |  |
| 10 | sae_top1_direction | relu | 1042.0 | 1 | -0.425 | 0.0266 | 0.098 |
| 10 | sae_top1_direction | relu | 1042.0 | -1 | -0.3293 | 0.1131 |  |
| 10 | sae_top10_direction | relu | 1042.0 | 1 | -0.3639 | 0.0632 | 0.1829 |
| 10 | sae_top10_direction | relu | 1042.0 | -1 | -0.4835 | 0.0033 |  |
| 10 | sae_reconstruction_direction | relu | 2042.0 | 1 | -0.077 | 0.0566 | 0.1797 |
| 10 | sae_reconstruction_direction | relu | 2042.0 | -1 | -0.5197 | 0.0033 |  |
| 10 | sae_top1_direction | relu | 2042.0 | 1 | -0.4192 | 0.0233 | 0.098 |
| 10 | sae_top1_direction | relu | 2042.0 | -1 | -0.326 | 0.0998 |  |
| 10 | sae_top10_direction | relu | 2042.0 | 1 | -0.4473 | 0.0133 | 0.077 |
| 10 | sae_top10_direction | relu | 2042.0 | -1 | -0.5421 | 0.0067 |  |
| 10 | sae_reconstruction_direction | topk | 42.0 | 1 | 0.0594 | 0.0033 | 0.03 |
| 10 | sae_reconstruction_direction | topk | 42.0 | -1 | -0.5617 | 0.0033 |  |
| 10 | sae_top1_direction | topk | 42.0 | 1 | 0.0016 | 0.0033 | 0.03 |
| 10 | sae_top1_direction | topk | 42.0 | -1 | -0.0987 | 0.1631 |  |
| 10 | sae_top10_direction | topk | 42.0 | 1 | -0.1181 | 0.3361 | 0.4862 |
| 10 | sae_top10_direction | topk | 42.0 | -1 | -0.3061 | 0.03 |  |
| 10 | sae_reconstruction_direction | topk | 1042.0 | 1 | 0.0273 | 0.0033 | 0.03 |
| 10 | sae_reconstruction_direction | topk | 1042.0 | -1 | -0.5489 | 0.0033 |  |
| 10 | sae_top1_direction | topk | 1042.0 | 1 | -0.17 | 0.8652 | 0.8871 |
| 10 | sae_top1_direction | topk | 1042.0 | -1 | -0.1763 | 0.782 |  |
| 10 | sae_top10_direction | topk | 1042.0 | 1 | -0.0886 | 0.0998 | 0.2378 |
| 10 | sae_top10_direction | topk | 1042.0 | -1 | -0.3252 | 0.0067 |  |
| 10 | sae_reconstruction_direction | topk | 2042.0 | 1 | 0.0236 | 0.0033 | 0.03 |
| 10 | sae_reconstruction_direction | topk | 2042.0 | -1 | -0.5478 | 0.0033 |  |
| 10 | sae_top1_direction | topk | 2042.0 | 1 | -0.0016 | 0.0033 | 0.03 |
| 10 | sae_top1_direction | topk | 2042.0 | -1 | -0.1924 | 0.5158 |  |
| 10 | sae_top10_direction | topk | 2042.0 | 1 | -0.1326 | 0.5125 | 0.6094 |
| 10 | sae_top10_direction | topk | 2042.0 | -1 | -0.3669 | 0.01 |  |
| 10 | sae_reconstruction_direction | jumprelu | 42.0 | 1 | 0.0236 | 0.0033 | 0.03 |
| 10 | sae_reconstruction_direction | jumprelu | 42.0 | -1 | -0.5399 | 0.0033 |  |
| 10 | sae_top1_direction | jumprelu | 42.0 | 1 | -0.2275 | 0.2496 | 0.438 |
| 10 | sae_top1_direction | jumprelu | 42.0 | -1 | -0.3038 | 0.0532 |  |
| 10 | sae_top10_direction | jumprelu | 42.0 | 1 | -0.1866 | 0.619 | 0.7038 |
| 10 | sae_top10_direction | jumprelu | 42.0 | -1 | -0.2598 | 0.0499 |  |
| 10 | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | 0.018 | 0.0033 | 0.03 |
| 10 | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | -0.5814 | 0.0033 |  |
| 10 | sae_top1_direction | jumprelu | 1042.0 | 1 | -0.1005 | 0.193 | 0.3813 |
| 10 | sae_top1_direction | jumprelu | 1042.0 | -1 | -0.1101 | 0.3095 |  |
| 10 | sae_top10_direction | jumprelu | 1042.0 | 1 | -0.254 | 0.0765 | 0.2067 |
| 10 | sae_top10_direction | jumprelu | 1042.0 | -1 | -0.3852 | 0.0033 |  |
| 10 | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | 0.0452 | 0.0033 | 0.03 |
| 10 | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | -0.539 | 0.0033 |  |
| 10 | sae_top1_direction | jumprelu | 2042.0 | 1 | -0.4127 | 0.02 | 0.0951 |
| 10 | sae_top1_direction | jumprelu | 2042.0 | -1 | -0.3176 | 0.0433 |  |
| 10 | sae_top10_direction | jumprelu | 2042.0 | 1 | -0.0617 | 0.0166 | 0.0899 |
| 10 | sae_top10_direction | jumprelu | 2042.0 | -1 | -0.2798 | 0.02 |  |

## Frozen-probe gap: SAE error vs random error of the same size (validation)

| run | domain | layer | architecture | logistic_auroc_gap | logistic_noise_gap_mean | logistic_excess_gap | logistic_signal_retention | logistic_noise_inflation | logistic_error_anisotropy | informative_layer |
|---|---|---|---|---|---|---|---|---|---|---|
| got_inhouse | geographic | 0 | jumprelu | 0.0387 | 0.0161 | 0.0225 | 0.0021 | 0.2813 | 1.4441 | 0.0 |
| got_inhouse | geographic | 0 | relu | 0.0386 | 0.0135 | 0.0251 | 0.0149 | 0.3432 | 1.0252 | 0.0 |
| got_inhouse | geographic | 0 | topk | 0.0217 | 0.0116 | 0.0101 | 0.185 | 0.4063 | 1.1779 | 0.0 |
| got_inhouse | geographic | 1 | jumprelu | 0.0391 | 0.0263 | 0.0129 | -1.2574 | 3.8652 | 0.1651 | 0.0 |
| got_inhouse | geographic | 1 | relu | 0.0401 | 0.0259 | 0.0142 | -0.5453 | 2.1539 | 0.2025 | 0.0 |
| got_inhouse | geographic | 1 | topk | 0.0282 | 0.0253 | 0.0029 | 0.3913 | 4.8392 | 0.5296 | 0.0 |
| got_inhouse | geographic | 2 | jumprelu | 0.0977 | 0.0881 | 0.0096 | -0.1003 | 4.4607 | 0.1748 | 0.0 |
| got_inhouse | geographic | 2 | relu | 0.1014 | 0.0873 | 0.0141 | -0.0878 | 3.8241 | 0.1525 | 0.0 |
| got_inhouse | geographic | 2 | topk | 0.099 | 0.0846 | 0.0144 | 0.0777 | 5.1861 | 0.4297 | 0.0 |
| got_inhouse | geographic | 3 | jumprelu | 0.144 | 0.1385 | 0.0055 | 0.2571 | 7.4487 | 0.2981 | 0.0 |
| got_inhouse | geographic | 3 | relu | 0.1488 | 0.1377 | 0.011 | -0.0504 | 5.9825 | 0.1997 | 0.0 |
| got_inhouse | geographic | 3 | topk | 0.1445 | 0.1343 | 0.0102 | 0.2347 | 5.4867 | 1.5198 | 0.0 |
| got_inhouse | geographic | 4 | jumprelu | 0.2044 | 0.1985 | 0.0059 | 0.18 | 6.0949 | 0.234 | 1.0 |
| got_inhouse | geographic | 4 | relu | 0.2037 | 0.1981 | 0.0056 | 0.2018 | 5.0155 | 0.209 | 1.0 |
| got_inhouse | geographic | 4 | topk | 0.1885 | 0.1939 | -0.0054 | 0.4692 | 6.2409 | 0.3145 | 1.0 |
| got_inhouse | geographic | 5 | jumprelu | 0.3134 | 0.2881 | 0.0252 | 0.0869 | 2.9297 | 0.1627 | 1.0 |
| got_inhouse | geographic | 5 | relu | 0.3046 | 0.2889 | 0.0157 | 0.1555 | 2.7515 | 0.2201 | 1.0 |
| got_inhouse | geographic | 5 | topk | 0.2746 | 0.2773 | -0.0026 | 0.4397 | 3.7703 | 0.4782 | 1.0 |
| got_inhouse | geographic | 6 | jumprelu | 0.3467 | 0.2953 | 0.0514 | 0.0048 | 1.9139 | 0.0978 | 1.0 |
| got_inhouse | geographic | 6 | relu | 0.3225 | 0.2985 | 0.024 | 0.1006 | 1.7788 | 0.0871 | 1.0 |
| got_inhouse | geographic | 6 | topk | 0.2803 | 0.2861 | -0.0057 | 0.3862 | 2.9972 | 0.5564 | 1.0 |
| got_inhouse | geographic | 7 | jumprelu | 0.3379 | 0.3102 | 0.0276 | 0.1954 | 2.5871 | 0.2355 | 1.0 |
| got_inhouse | geographic | 7 | relu | 0.3424 | 0.3114 | 0.031 | 0.0795 | 1.3104 | 0.2056 | 1.0 |
| got_inhouse | geographic | 7 | topk | 0.2889 | 0.3001 | -0.0111 | 0.3152 | 1.8296 | 0.7083 | 1.0 |
| got_inhouse | geographic | 8 | jumprelu | 0.2918 | 0.2961 | -0.0042 | 0.3497 | 2.0442 | 0.2762 | 1.0 |
| got_inhouse | geographic | 8 | relu | 0.2931 | 0.297 | -0.0039 | 0.1182 | 1.0812 | 0.0895 | 1.0 |
| got_inhouse | geographic | 8 | topk | 0.2897 | 0.2859 | 0.0038 | 0.3085 | 2.0472 | 0.3061 | 1.0 |
| got_inhouse | geographic | 9 | jumprelu | 0.3411 | 0.2702 | 0.0709 | 0.0876 | 1.5894 | 0.3752 | 1.0 |
| got_inhouse | geographic | 9 | relu | 0.3199 | 0.2687 | 0.0512 | 0.1275 | 1.1387 | 0.2595 | 1.0 |
| got_inhouse | geographic | 9 | topk | 0.3206 | 0.2554 | 0.0651 | 0.18 | 2.3264 | 1.2284 | 1.0 |
| got_inhouse | geographic | 10 | jumprelu | 0.2892 | 0.246 | 0.0432 | 0.3126 | 2.1138 | 0.5673 | 1.0 |
| got_inhouse | geographic | 10 | relu | 0.2976 | 0.2509 | 0.0467 | 0.2122 | 1.586 | 0.4551 | 1.0 |
| got_inhouse | geographic | 10 | topk | 0.2683 | 0.2311 | 0.0372 | 0.4298 | 2.1149 | 0.9335 | 1.0 |
| got_inhouse | geographic | 11 | jumprelu | 0.2675 | 0.2253 | 0.0422 | 0.3406 | 1.9993 | 0.3871 | 1.0 |
| got_inhouse | geographic | 11 | relu | 0.2518 | 0.2332 | 0.0186 | 0.3111 | 1.5833 | 0.3771 | 1.0 |
| got_inhouse | geographic | 11 | topk | 0.2365 | 0.2151 | 0.0214 | 0.5361 | 2.4019 | 0.9174 | 1.0 |
| got_inhouse | numerical | 0 | jumprelu | 0.0 | 0.0 | 0.0 |  |  |  | 0.0 |
| got_inhouse | numerical | 0 | relu | 0.0 | 0.0 | 0.0 |  |  |  | 0.0 |
| got_inhouse | numerical | 0 | topk | 0.0 | 0.0 | 0.0 |  |  |  | 0.0 |
| got_inhouse | numerical | 1 | jumprelu | 0.4889 | 0.4839 | 0.005 | 0.0527 | 6.9029 | 0.5369 | 1.0 |
| got_inhouse | numerical | 1 | relu | 0.47 | 0.4811 | -0.0111 | 0.1874 | 6.0651 | 0.0882 | 1.0 |
| got_inhouse | numerical | 1 | topk | 0.4701 | 0.4753 | -0.0052 | 0.3423 | 8.7533 | 0.4162 | 1.0 |
| got_inhouse | numerical | 2 | jumprelu | 0.4809 | 0.4812 | -0.0003 | 0.1507 | 8.1116 | 1.2544 | 1.0 |
| got_inhouse | numerical | 2 | relu | 0.488 | 0.4791 | 0.0089 | 0.0564 | 4.8598 | 1.3161 | 1.0 |
| got_inhouse | numerical | 2 | topk | 0.4697 | 0.4733 | -0.0036 | 0.3533 | 10.165 | 0.1963 | 1.0 |
| got_inhouse | numerical | 3 | jumprelu | 0.4958 | 0.4786 | 0.0173 | 0.0447 | 8.0962 | 0.7438 | 1.0 |
| got_inhouse | numerical | 3 | relu | 0.4859 | 0.4779 | 0.008 | 0.053 | 4.9548 | 1.9179 | 1.0 |
| got_inhouse | numerical | 3 | topk | 0.4802 | 0.4721 | 0.0081 | 0.2096 | 7.2488 | 1.3143 | 1.0 |
| got_inhouse | numerical | 4 | jumprelu | 0.4529 | 0.477 | -0.0242 | 0.3873 | 6.332 | 1.7905 | 1.0 |
| got_inhouse | numerical | 4 | relu | 0.4834 | 0.4766 | 0.0068 | 0.0934 | 4.5488 | 1.8423 | 1.0 |
| got_inhouse | numerical | 4 | topk | 0.4651 | 0.4715 | -0.0063 | 0.2197 | 6.3119 | 1.2618 | 1.0 |
| got_inhouse | numerical | 5 | jumprelu | 0.4699 | 0.4736 | -0.0037 | 0.1948 | 5.5767 | 0.0635 | 1.0 |
| got_inhouse | numerical | 5 | relu | 0.4635 | 0.4734 | -0.0098 | 0.1555 | 3.3549 | 0.0589 | 1.0 |
| got_inhouse | numerical | 5 | topk | 0.4563 | 0.467 | -0.0107 | 0.2872 | 5.4617 | 1.7907 | 1.0 |
| got_inhouse | numerical | 6 | jumprelu | 0.4774 | 0.4603 | 0.0171 | 0.1001 | 3.8684 | 2.843 | 1.0 |
| got_inhouse | numerical | 6 | relu | 0.4554 | 0.4606 | -0.0052 | 0.1379 | 2.2696 | 0.5281 | 1.0 |
| got_inhouse | numerical | 6 | topk | 0.4332 | 0.4514 | -0.0183 | 0.3596 | 3.9909 | 2.7308 | 1.0 |
| got_inhouse | numerical | 7 | jumprelu | 0.4395 | 0.4559 | -0.0164 | 0.324 | 4.021 | 0.3825 | 1.0 |
| got_inhouse | numerical | 7 | relu | 0.4271 | 0.4561 | -0.029 | 0.1978 | 2.0733 | 0.1271 | 1.0 |
| got_inhouse | numerical | 7 | topk | 0.4542 | 0.4483 | 0.0058 | 0.275 | 4.6331 | 0.3612 | 1.0 |
| got_inhouse | numerical | 8 | jumprelu | 0.4588 | 0.4464 | 0.0124 | 0.1846 | 3.2296 | 0.3858 | 1.0 |
| got_inhouse | numerical | 8 | relu | 0.4652 | 0.4485 | 0.0167 | 0.1941 | 3.3811 | 1.0038 | 1.0 |
| got_inhouse | numerical | 8 | topk | 0.4682 | 0.4388 | 0.0295 | 0.1713 | 4.0427 | 1.8922 | 1.0 |
| got_inhouse | numerical | 9 | jumprelu | 0.4328 | 0.4492 | -0.0164 | 0.2952 | 3.0594 | 0.5593 | 1.0 |
| got_inhouse | numerical | 9 | relu | 0.4454 | 0.4508 | -0.0054 | 0.1654 | 2.3495 | 0.1268 | 1.0 |
| got_inhouse | numerical | 9 | topk | 0.4258 | 0.4404 | -0.0146 | 0.347 | 3.9051 | 0.3493 | 1.0 |
| got_inhouse | numerical | 10 | jumprelu | 0.4573 | 0.4387 | 0.0186 | 0.1956 | 4.1035 | 0.2977 | 1.0 |
| got_inhouse | numerical | 10 | relu | 0.4712 | 0.4435 | 0.0278 | 0.1326 | 3.6245 | 0.2069 | 1.0 |
| got_inhouse | numerical | 10 | topk | 0.4391 | 0.43 | 0.0091 | 0.3391 | 4.6812 | 1.3571 | 1.0 |
| got_inhouse | numerical | 11 | jumprelu | 0.5044 | 0.4394 | 0.0649 | 0.0007 | 2.9503 | 0.6733 | 1.0 |
| got_inhouse | numerical | 11 | relu | 0.4336 | 0.4446 | -0.011 | 0.1714 | 2.2228 | 0.1844 | 1.0 |
| got_inhouse | numerical | 11 | topk | 0.4335 | 0.4303 | 0.0033 | 0.3229 | 3.6568 | 0.6798 | 1.0 |
| got_public_relu | geographic | 0 | public_relu | 0.042 | 0.0092 | 0.0327 | -0.0058 | 0.3362 | 1.6785 | 0.0 |
| got_public_relu | geographic | 1 | public_relu | 0.0263 | 0.0215 | 0.0048 | 0.1749 | 1.983 | 0.0367 | 0.0 |
| got_public_relu | geographic | 2 | public_relu | 0.0944 | 0.0836 | 0.0109 | 0.0308 | 2.1468 | 0.1035 | 0.0 |
| got_public_relu | geographic | 3 | public_relu | 0.1454 | 0.1415 | 0.004 | -0.0493 | 4.6055 | 0.1363 | 0.0 |
| got_public_relu | geographic | 4 | public_relu | 0.2102 | 0.1913 | 0.0189 | 0.0881 | 2.6792 | 0.5764 | 1.0 |
| got_public_relu | geographic | 5 | public_relu | 0.3094 | 0.2761 | 0.0334 | 0.1293 | 2.6802 | 0.1494 | 1.0 |
| got_public_relu | geographic | 6 | public_relu | 0.3236 | 0.2946 | 0.029 | 0.1238 | 2.4611 | 0.1613 | 1.0 |
| got_public_relu | geographic | 7 | public_relu | 0.341 | 0.2944 | 0.0466 | 0.0855 | 2.0623 | 0.158 | 1.0 |
| got_public_relu | geographic | 8 | public_relu | 0.3124 | 0.286 | 0.0264 | 0.085 | 0.9112 | 0.1978 | 1.0 |
| got_public_relu | geographic | 9 | public_relu | 0.3318 | 0.2652 | 0.0666 | 0.0868 | 1.5409 | 0.754 | 1.0 |
| got_public_relu | geographic | 10 | public_relu | 0.3075 | 0.2429 | 0.0647 | 0.1055 | 0.8567 | 0.2762 | 1.0 |
| got_public_relu | geographic | 11 | public_relu | 0.3123 | 0.2226 | 0.0897 | 0.0635 | 1.3718 | 0.4212 | 1.0 |
| got_public_relu | numerical | 0 | public_relu | 0.0 | 0.0 | 0.0 |  |  |  | 0.0 |
| got_public_relu | numerical | 1 | public_relu | 0.4806 | 0.4789 | 0.0017 | 0.0824 | 4.0291 | 0.0045 | 1.0 |
| got_public_relu | numerical | 2 | public_relu | 0.4844 | 0.4753 | 0.0091 | 0.0439 | 4.1687 | 1.0224 | 1.0 |
| got_public_relu | numerical | 3 | public_relu | 0.4921 | 0.4736 | 0.0185 | 0.0108 | 5.5124 | 1.3801 | 1.0 |
| got_public_relu | numerical | 4 | public_relu | 0.4806 | 0.4781 | 0.0025 | 0.0532 | 3.2882 | 4.1813 | 1.0 |
| got_public_relu | numerical | 5 | public_relu | 0.4706 | 0.4683 | 0.0023 | 0.097 | 3.9145 | 1.4877 | 1.0 |
| got_public_relu | numerical | 6 | public_relu | 0.4855 | 0.4507 | 0.0349 | 0.0349 | 2.6809 | 0.7509 | 1.0 |
| got_public_relu | numerical | 7 | public_relu | 0.4348 | 0.4493 | -0.0145 | 0.1729 | 2.196 | 0.7936 | 1.0 |
| got_public_relu | numerical | 8 | public_relu | 0.5041 | 0.4367 | 0.0675 | 0.0158 | 1.8946 | 1.1969 | 1.0 |
| got_public_relu | numerical | 9 | public_relu | 0.4782 | 0.4433 | 0.0349 | 0.0493 | 2.6047 | 0.4256 | 1.0 |
| got_public_relu | numerical | 10 | public_relu | 0.4926 | 0.4438 | 0.0487 | 0.0018 | 1.9206 | 0.727 | 1.0 |
| got_public_relu | numerical | 11 | public_relu | 0.4848 | 0.4388 | 0.0459 | 0.0848 | 1.843 | 0.9975 | 1.0 |
| got_public_topk | geographic | 1 | public_topk | 0.0273 | 0.0211 | 0.0062 | -0.8029 | 4.1958 | 0.2489 | 0.0 |
| got_public_topk | geographic | 2 | public_topk | 0.0852 | 0.0836 | 0.0016 | 0.3392 | 4.4177 | 0.1085 | 0.0 |
| got_public_topk | geographic | 3 | public_topk | 0.1464 | 0.1403 | 0.006 | 0.2195 | 6.4347 | 0.5628 | 0.0 |
| got_public_topk | geographic | 4 | public_topk | 0.2199 | 0.1899 | 0.0301 | -0.2457 | 6.7532 | 0.4611 | 1.0 |
| got_public_topk | geographic | 5 | public_topk | 0.2891 | 0.2756 | 0.0135 | 0.2674 | 2.2189 | 0.0949 | 1.0 |
| got_public_topk | geographic | 6 | public_topk | 0.2635 | 0.2959 | -0.0325 | 0.4339 | 2.1064 | 0.2526 | 1.0 |
| got_public_topk | geographic | 7 | public_topk | 0.3806 | 0.3009 | 0.0798 | -0.0515 | 2.0205 | 0.1259 | 1.0 |
| got_public_topk | geographic | 8 | public_topk | 0.3244 | 0.2896 | 0.0348 | 0.0865 | 1.3687 | 0.203 | 1.0 |
| got_public_topk | geographic | 9 | public_topk | 0.3416 | 0.276 | 0.0656 | 0.09 | 2.4125 | 1.6858 | 1.0 |
| got_public_topk | geographic | 10 | public_topk | 0.3023 | 0.2421 | 0.0602 | 0.2506 | 1.6185 | 0.4252 | 1.0 |
| got_public_topk | geographic | 11 | public_topk | 0.2713 | 0.219 | 0.0523 | 0.4169 | 2.7679 | 1.1854 | 1.0 |
| got_public_topk | numerical | 1 | public_topk | 0.5104 | 0.4741 | 0.0364 | -0.4145 | 13.7397 | 0.502 | 1.0 |
| got_public_topk | numerical | 2 | public_topk | 0.4924 | 0.4739 | 0.0184 | 0.0243 | 10.8712 | 0.3038 | 1.0 |
| got_public_topk | numerical | 3 | public_topk | 0.4818 | 0.4695 | 0.0123 | 0.0339 | 8.6351 | 0.3097 | 1.0 |
| got_public_topk | numerical | 4 | public_topk | 0.4598 | 0.4743 | -0.0144 | 0.2924 | 6.5161 | 1.4267 | 1.0 |
| got_public_topk | numerical | 5 | public_topk | 0.4188 | 0.4691 | -0.0504 | 0.5322 | 5.7908 | 0.0959 | 1.0 |
| got_public_topk | numerical | 6 | public_topk | 0.4862 | 0.4489 | 0.0373 | -0.0019 | 3.057 | 0.6792 | 1.0 |
| got_public_topk | numerical | 7 | public_topk | 0.5045 | 0.4464 | 0.058 | 0.021 | 3.3017 | 1.631 | 1.0 |
| got_public_topk | numerical | 8 | public_topk | 0.4524 | 0.4286 | 0.0238 | 0.1723 | 3.3256 | 0.4539 | 1.0 |
| got_public_topk | numerical | 9 | public_topk | 0.4424 | 0.4394 | 0.003 | 0.1922 | 2.5282 | 0.3126 | 1.0 |
| got_public_topk | numerical | 10 | public_topk | 0.4643 | 0.4312 | 0.0331 | 0.1947 | 2.7265 | 1.227 | 1.0 |
| got_public_topk | numerical | 11 | public_topk | 0.4649 | 0.4208 | 0.0441 | 0.2002 | 3.3634 | 0.1406 | 1.0 |
| repe_inhouse | honesty | 0 | jumprelu | 0.0135 | 0.0595 | -0.046 | 1.0117 | 1.0698 | 0.2272 | 1.0 |
| repe_inhouse | honesty | 0 | relu | 0.0111 | 0.027 | -0.016 | 0.9864 | 1.024 | 0.3933 | 1.0 |
| repe_inhouse | honesty | 0 | topk | 0.0063 | 0.0428 | -0.0365 | 0.9918 | 1.0123 | 0.211 | 1.0 |
| repe_inhouse | honesty | 1 | jumprelu | 0.0371 | 0.0326 | 0.0046 | 0.8727 | 1.0913 | 1.0909 | 1.0 |
| repe_inhouse | honesty | 1 | relu | 0.0529 | 0.0358 | 0.0171 | 0.913 | 1.2249 | 1.2716 | 1.0 |
| repe_inhouse | honesty | 1 | topk | 0.0521 | 0.0473 | 0.0049 | 0.8686 | 1.1568 | 1.0843 | 1.0 |
| repe_inhouse | honesty | 2 | jumprelu | 0.0414 | 0.0398 | 0.0016 | 0.8104 | 1.1226 | 0.844 | 1.0 |
| repe_inhouse | honesty | 2 | relu | 0.0588 | 0.0467 | 0.0121 | 0.8645 | 1.2825 | 1.0708 | 1.0 |
| repe_inhouse | honesty | 2 | topk | 0.0484 | 0.058 | -0.0097 | 0.8144 | 1.1464 | 0.8624 | 1.0 |
| repe_inhouse | honesty | 3 | jumprelu | 0.0583 | 0.0372 | 0.0212 | 0.791 | 1.3926 | 1.8605 | 1.0 |
| repe_inhouse | honesty | 3 | relu | 0.1288 | 0.0502 | 0.0787 | 0.792 | 1.9413 | 1.6521 | 1.0 |
| repe_inhouse | honesty | 3 | topk | 0.0694 | 0.0544 | 0.0149 | 0.7678 | 1.4162 | 1.4562 | 1.0 |
| repe_inhouse | honesty | 4 | jumprelu | 0.0671 | 0.0582 | 0.0089 | 0.7803 | 1.4258 | 1.2495 | 1.0 |
| repe_inhouse | honesty | 4 | relu | 0.1546 | 0.0777 | 0.0769 | 0.8049 | 2.1812 | 1.6485 | 1.0 |
| repe_inhouse | honesty | 4 | topk | 0.091 | 0.0762 | 0.0148 | 0.7487 | 1.5157 | 1.2251 | 1.0 |
| repe_inhouse | honesty | 5 | jumprelu | 0.0569 | 0.0863 | -0.0294 | 0.7971 | 1.4042 | 1.3587 | 1.0 |
| repe_inhouse | honesty | 5 | relu | 0.2181 | 0.1115 | 0.1066 | 0.7828 | 2.86 | 1.9327 | 1.0 |
| repe_inhouse | honesty | 5 | topk | 0.0925 | 0.1054 | -0.0129 | 0.7612 | 1.5743 | 1.361 | 1.0 |
| repe_inhouse | honesty | 6 | jumprelu | 0.062 | 0.0963 | -0.0343 | 0.7981 | 1.5597 | 1.0335 | 1.0 |
| repe_inhouse | honesty | 6 | relu | 0.2457 | 0.1252 | 0.1206 | 0.7501 | 3.3753 | 2.8579 | 1.0 |
| repe_inhouse | honesty | 6 | topk | 0.0944 | 0.1153 | -0.0209 | 0.7432 | 1.6845 | 1.0946 | 1.0 |
| repe_inhouse | honesty | 7 | jumprelu | 0.0884 | 0.1244 | -0.036 | 0.751 | 1.5743 | 1.1813 | 1.0 |
| repe_inhouse | honesty | 7 | relu | 0.2853 | 0.1547 | 0.1306 | 0.7122 | 3.7169 | 2.5396 | 1.0 |
| repe_inhouse | honesty | 7 | topk | 0.0881 | 0.1428 | -0.0547 | 0.7608 | 1.5969 | 0.8642 | 1.0 |
| repe_inhouse | honesty | 8 | jumprelu | 0.1111 | 0.1539 | -0.0429 | 0.7631 | 1.8011 | 0.7114 | 1.0 |
| repe_inhouse | honesty | 8 | relu | 0.2856 | 0.1818 | 0.1038 | 0.7534 | 3.8239 | 2.1905 | 1.0 |
| repe_inhouse | honesty | 8 | topk | 0.0772 | 0.1669 | -0.0896 | 0.7904 | 1.5793 | 0.4547 | 1.0 |
| repe_inhouse | honesty | 9 | jumprelu | 0.1441 | 0.1857 | -0.0417 | 0.7493 | 1.9859 | 0.4755 | 1.0 |
| repe_inhouse | honesty | 9 | relu | 0.3171 | 0.2091 | 0.108 | 0.7016 | 4.2645 | 2.1832 | 1.0 |
| repe_inhouse | honesty | 9 | topk | 0.1313 | 0.195 | -0.0637 | 0.6893 | 1.7386 | 0.3806 | 1.0 |
| repe_inhouse | honesty | 10 | jumprelu | 0.2233 | 0.2181 | 0.0052 | 0.696 | 2.47 | 0.6242 | 1.0 |
| repe_inhouse | honesty | 10 | relu | 0.3493 | 0.2391 | 0.1102 | 0.5597 | 3.9218 | 1.6366 | 1.0 |
| repe_inhouse | honesty | 10 | topk | 0.1784 | 0.2227 | -0.0443 | 0.6716 | 2.0921 | 0.4461 | 1.0 |
| repe_inhouse | honesty | 11 | jumprelu | 0.216 | 0.2171 | -0.0011 | 0.6257 | 2.0622 | 0.5647 | 1.0 |
| repe_inhouse | honesty | 11 | relu | 0.3396 | 0.2404 | 0.0992 | 0.509 | 3.0499 | 1.4353 | 1.0 |
| repe_inhouse | honesty | 11 | topk | 0.1526 | 0.2195 | -0.0669 | 0.6142 | 1.433 | 0.3272 | 1.0 |

## Matched-L0 comparison (got, logistic gap, informative layers)

| domain | layer | m | comparison | mean_diff | sd_diff | all_seeds_agree |
|---|---|---|---|---|---|---|
| geographic | 4 | 8.0 | topk - relu | 0.0252 | 0.0199 | a_higher |
| geographic | 4 | 8.0 | topk - jumprelu | 0.0175 | 0.0468 | mixed |
| geographic | 4 | 8.0 | jumprelu - relu | 0.0077 | 0.029 | mixed |
| geographic | 4 | 16.0 | topk - relu | 0.0171 | 0.0446 | mixed |
| geographic | 4 | 16.0 | topk - jumprelu | 0.0089 | 0.0518 | mixed |
| geographic | 4 | 16.0 | jumprelu - relu | 0.0082 | 0.011 | mixed |
| geographic | 4 | 24.0 | topk - relu | 0.0049 | 0.0086 | mixed |
| geographic | 4 | 24.0 | topk - jumprelu | 0.0151 | 0.038 | mixed |
| geographic | 4 | 24.0 | jumprelu - relu | -0.0102 | 0.0304 | mixed |
| geographic | 4 | 32.0 | topk - relu | 0.0143 | 0.0043 | a_higher |
| geographic | 4 | 32.0 | topk - jumprelu | 0.002 | 0.007 | mixed |
| geographic | 4 | 32.0 | jumprelu - relu | 0.0123 | 0.011 | a_higher |
| geographic | 5 | 8.0 | topk - relu | 0.0133 | 0.0159 | a_higher |
| geographic | 5 | 8.0 | topk - jumprelu | 0.0219 | 0.0056 | a_higher |
| geographic | 5 | 8.0 | jumprelu - relu | -0.0086 | 0.0204 | mixed |
| geographic | 5 | 16.0 | topk - relu | 0.0066 | 0.024 | mixed |
| geographic | 5 | 16.0 | topk - jumprelu | -0.0053 | 0.0744 | mixed |
| geographic | 5 | 16.0 | jumprelu - relu | 0.0119 | 0.0733 | mixed |
| geographic | 5 | 24.0 | topk - relu | 0.0052 | 0.0329 | mixed |
| geographic | 5 | 24.0 | topk - jumprelu | -0.0085 | 0.0055 | a_lower |
| geographic | 5 | 24.0 | jumprelu - relu | 0.0137 | 0.0303 | mixed |
| geographic | 5 | 32.0 | topk - relu | 0.0205 | 0.0441 | mixed |
| geographic | 5 | 32.0 | topk - jumprelu | 0.0185 | 0.0276 | mixed |
| geographic | 5 | 32.0 | jumprelu - relu | 0.002 | 0.0166 | mixed |
| geographic | 6 | 8.0 | topk - relu | -0.0005 | 0.0553 | mixed |
| geographic | 6 | 8.0 | topk - jumprelu | 0.0208 | 0.0423 | mixed |
| geographic | 6 | 8.0 | jumprelu - relu | -0.0213 | 0.025 | mixed |
| geographic | 6 | 16.0 | topk - relu | -0.0248 | 0.014 | a_lower |
| geographic | 6 | 16.0 | topk - jumprelu | -0.0118 | 0.0158 | a_lower |
| geographic | 6 | 16.0 | jumprelu - relu | -0.0129 | 0.0097 | a_lower |
| geographic | 6 | 24.0 | topk - relu | 0.0125 | 0.0286 | mixed |
| geographic | 6 | 24.0 | topk - jumprelu | 0.0017 | 0.0656 | mixed |
| geographic | 6 | 24.0 | jumprelu - relu | 0.0108 | 0.0419 | mixed |
| geographic | 6 | 32.0 | topk - relu | 0.0033 | 0.0324 | mixed |
| geographic | 6 | 32.0 | topk - jumprelu | 0.0244 | 0.0572 | mixed |
| geographic | 6 | 32.0 | jumprelu - relu | -0.0211 | 0.0314 | mixed |
| geographic | 7 | 8.0 | topk - relu | 0.0145 | 0.0294 | mixed |
| geographic | 7 | 8.0 | topk - jumprelu | -0.0011 | 0.0439 | mixed |
| geographic | 7 | 8.0 | jumprelu - relu | 0.0156 | 0.0149 | a_higher |
| geographic | 7 | 16.0 | topk - relu | -0.0191 | 0.0123 | a_lower |
| geographic | 7 | 16.0 | topk - jumprelu | -0.0001 | 0.0158 | mixed |
| geographic | 7 | 16.0 | jumprelu - relu | -0.019 | 0.0158 | a_lower |
| geographic | 7 | 24.0 | topk - relu | 0.0143 | 0.0195 | mixed |
| geographic | 7 | 24.0 | topk - jumprelu | 0.0038 | 0.0118 | mixed |
| geographic | 7 | 24.0 | jumprelu - relu | 0.0105 | 0.0302 | mixed |
| geographic | 7 | 32.0 | topk - relu | -0.009 | 0.0415 | mixed |
| geographic | 7 | 32.0 | topk - jumprelu | -0.0106 | 0.0697 | mixed |
| geographic | 7 | 32.0 | jumprelu - relu | 0.0016 | 0.0285 | mixed |
| geographic | 7 | 48.0 | topk - relu | 0.0035 | 0.0278 | mixed |
| geographic | 7 | 48.0 | topk - jumprelu | -0.0174 | 0.0046 | a_lower |
| geographic | 7 | 48.0 | jumprelu - relu | 0.0209 | 0.0233 | mixed |
| geographic | 7 | 64.0 | topk - relu | -0.0081 | 0.0056 | a_lower |
| geographic | 7 | 64.0 | topk - jumprelu | -0.0019 | 0.0115 | mixed |
| geographic | 7 | 64.0 | jumprelu - relu | -0.0062 | 0.0104 | mixed |
| geographic | 8 | 8.0 | topk - relu | -0.009 | 0.0493 | mixed |
| geographic | 8 | 8.0 | topk - jumprelu | 0.0161 | 0.0473 | mixed |
| geographic | 8 | 8.0 | jumprelu - relu | -0.0251 | 0.0146 | a_lower |
| geographic | 8 | 16.0 | topk - relu | 0.0254 | 0.0523 | mixed |
| geographic | 8 | 16.0 | topk - jumprelu | 0.0263 | 0.0481 | mixed |
| geographic | 8 | 16.0 | jumprelu - relu | -0.0009 | 0.0442 | mixed |
| geographic | 8 | 24.0 | topk - relu | -0.0133 | 0.0462 | mixed |
| geographic | 8 | 24.0 | topk - jumprelu | -0.0555 | 0.0287 | a_lower |
| geographic | 8 | 24.0 | jumprelu - relu | 0.0421 | 0.0176 | a_higher |
| geographic | 8 | 32.0 | topk - relu | 0.0128 | 0.0612 | mixed |
| geographic | 8 | 32.0 | topk - jumprelu | 0.0018 | 0.0235 | mixed |
| geographic | 8 | 32.0 | jumprelu - relu | 0.011 | 0.0377 | mixed |
| geographic | 8 | 48.0 | topk - relu | 0.0271 | 0.0876 | mixed |
| geographic | 8 | 48.0 | topk - jumprelu | 0.0061 | 0.0383 | mixed |
| geographic | 8 | 48.0 | jumprelu - relu | 0.0209 | 0.0508 | mixed |
| geographic | 8 | 64.0 | topk - relu | 0.0109 | 0.113 | mixed |
| geographic | 8 | 64.0 | topk - jumprelu | 0.0071 | 0.0682 | mixed |
| geographic | 8 | 64.0 | jumprelu - relu | 0.0038 | 0.0487 | mixed |
| geographic | 9 | 8.0 | topk - relu | 0.0112 | 0.0116 | a_higher |
| geographic | 9 | 8.0 | topk - jumprelu | 0.016 | 0.0167 | mixed |
| geographic | 9 | 8.0 | jumprelu - relu | -0.0048 | 0.027 | mixed |
| geographic | 9 | 16.0 | topk - relu | 0.0277 | 0.0538 | mixed |
| geographic | 9 | 16.0 | topk - jumprelu | -0.0139 | 0.0337 | mixed |
| geographic | 9 | 16.0 | jumprelu - relu | 0.0415 | 0.0396 | mixed |
| geographic | 9 | 24.0 | topk - relu | 0.0217 | 0.0429 | mixed |
| geographic | 9 | 24.0 | topk - jumprelu | 0.0258 | 0.0362 | mixed |
| geographic | 9 | 24.0 | jumprelu - relu | -0.0041 | 0.057 | mixed |
| geographic | 9 | 32.0 | topk - relu | -0.0025 | 0.035 | mixed |
| geographic | 9 | 32.0 | topk - jumprelu | 0.027 | 0.053 | mixed |
| geographic | 9 | 32.0 | jumprelu - relu | -0.0295 | 0.0315 | mixed |
| geographic | 9 | 48.0 | topk - relu | 0.0045 | 0.0252 | mixed |
| geographic | 9 | 48.0 | topk - jumprelu | 0.0001 | 0.0224 | mixed |
| geographic | 9 | 48.0 | jumprelu - relu | 0.0044 | 0.0274 | mixed |
| geographic | 9 | 64.0 | topk - relu | 0.0064 | 0.0458 | mixed |
| geographic | 9 | 64.0 | topk - jumprelu | -0.0113 | 0.0354 | mixed |
| geographic | 9 | 64.0 | jumprelu - relu | 0.0177 | 0.0244 | mixed |
| geographic | 10 | 8.0 | topk - relu | -0.0423 | 0.0431 | a_lower |
| geographic | 10 | 8.0 | topk - jumprelu | 0.0023 | 0.0206 | mixed |
| geographic | 10 | 8.0 | jumprelu - relu | -0.0446 | 0.0349 | a_lower |
| geographic | 10 | 16.0 | topk - relu | 0.0188 | 0.0468 | mixed |
| geographic | 10 | 16.0 | topk - jumprelu | 0.0348 | 0.0219 | a_higher |
| geographic | 10 | 16.0 | jumprelu - relu | -0.016 | 0.038 | mixed |
| geographic | 10 | 24.0 | topk - relu | 0.0353 | 0.042 | mixed |
| geographic | 10 | 24.0 | topk - jumprelu | 0.0349 | 0.0481 | mixed |
| geographic | 10 | 24.0 | jumprelu - relu | 0.0004 | 0.0204 | mixed |
| geographic | 10 | 32.0 | topk - relu | 0.015 | 0.0544 | mixed |
| geographic | 10 | 32.0 | topk - jumprelu | 0.0161 | 0.0371 | mixed |
| geographic | 10 | 32.0 | jumprelu - relu | -0.0011 | 0.026 | mixed |
| geographic | 10 | 48.0 | topk - relu | -0.0212 | 0.0214 | a_lower |
| geographic | 10 | 48.0 | topk - jumprelu | -0.0186 | 0.0112 | a_lower |
| geographic | 10 | 48.0 | jumprelu - relu | -0.0026 | 0.0103 | mixed |
| geographic | 10 | 64.0 | topk - relu | -0.0331 | 0.0287 | a_lower |
| geographic | 10 | 64.0 | topk - jumprelu | -0.0302 | 0.0097 | a_lower |
| geographic | 10 | 64.0 | jumprelu - relu | -0.0029 | 0.0224 | mixed |
| geographic | 11 | 8.0 | topk - relu | -0.0189 | 0.0096 | a_lower |
| geographic | 11 | 8.0 | topk - jumprelu | -0.0013 | 0.0039 | mixed |
| geographic | 11 | 8.0 | jumprelu - relu | -0.0176 | 0.0105 | a_lower |
| geographic | 11 | 16.0 | topk - relu | -0.0255 | 0.0072 | a_lower |
| geographic | 11 | 16.0 | topk - jumprelu | 0.011 | 0.018 | mixed |
| geographic | 11 | 16.0 | jumprelu - relu | -0.0365 | 0.0121 | a_lower |
| geographic | 11 | 24.0 | topk - relu | 0.0285 | 0.038 | mixed |
| geographic | 11 | 24.0 | topk - jumprelu | 0.0157 | 0.0362 | mixed |
| geographic | 11 | 24.0 | jumprelu - relu | 0.0128 | 0.0134 | a_higher |
| geographic | 11 | 32.0 | topk - relu | 0.0311 | 0.0125 | a_higher |
| geographic | 11 | 32.0 | topk - jumprelu | 0.0023 | 0.0248 | mixed |
| geographic | 11 | 32.0 | jumprelu - relu | 0.0288 | 0.0173 | a_higher |
| geographic | 11 | 48.0 | topk - relu | 0.0038 | 0.0137 | mixed |
| geographic | 11 | 48.0 | topk - jumprelu | -0.0232 | 0.0095 | a_lower |
| geographic | 11 | 48.0 | jumprelu - relu | 0.027 | 0.0091 | a_higher |
| geographic | 11 | 64.0 | topk - relu | 0.0079 | 0.0487 | mixed |
| geographic | 11 | 64.0 | topk - jumprelu | 0.0041 | 0.0298 | mixed |
| geographic | 11 | 64.0 | jumprelu - relu | 0.0039 | 0.0273 | mixed |
| geographic | 11 | 96.0 | topk - relu | -0.0104 | 0.0294 | mixed |
| geographic | 11 | 96.0 | topk - jumprelu | -0.0278 | 0.0101 | a_lower |
| geographic | 11 | 96.0 | jumprelu - relu | 0.0174 | 0.0203 | a_higher |
| numerical | 1 | 8.0 | topk - relu | 0.0168 | 0.0134 | a_higher |
| numerical | 1 | 8.0 | topk - jumprelu | 0.006 | 0.0086 | mixed |
| numerical | 1 | 8.0 | jumprelu - relu | 0.0108 | 0.0059 | a_higher |
| numerical | 1 | 16.0 | topk - relu | -0.0008 | 0.0366 | mixed |
| numerical | 1 | 16.0 | topk - jumprelu | -0.0282 | 0.0262 | a_lower |
| numerical | 1 | 16.0 | jumprelu - relu | 0.0273 | 0.0164 | a_higher |
| numerical | 2 | 8.0 | topk - relu | -0.0114 | 0.0129 | a_lower |
| numerical | 2 | 8.0 | topk - jumprelu | -0.0086 | 0.0131 | a_lower |
| numerical | 2 | 8.0 | jumprelu - relu | -0.0028 | 0.0041 | mixed |
| numerical | 2 | 16.0 | topk - relu | 0.0075 | 0.0285 | mixed |
| numerical | 2 | 16.0 | topk - jumprelu | -0.0064 | 0.0247 | mixed |
| numerical | 2 | 16.0 | jumprelu - relu | 0.0139 | 0.0208 | mixed |
| numerical | 2 | 24.0 | topk - relu | -0.0011 | 0.0041 | mixed |
| numerical | 2 | 24.0 | topk - jumprelu | -0.004 | 0.0254 | mixed |
| numerical | 2 | 24.0 | jumprelu - relu | 0.0029 | 0.0213 | mixed |
| numerical | 3 | 8.0 | topk - relu | -0.0046 | 0.022 | mixed |
| numerical | 3 | 8.0 | topk - jumprelu | -0.0066 | 0.0036 | a_lower |
| numerical | 3 | 8.0 | jumprelu - relu | 0.0019 | 0.0256 | mixed |
| numerical | 3 | 16.0 | topk - relu | 0.0142 | 0.0069 | a_higher |
| numerical | 3 | 16.0 | topk - jumprelu | 0.0053 | 0.026 | mixed |
| numerical | 3 | 16.0 | jumprelu - relu | 0.0088 | 0.0232 | mixed |
| numerical | 3 | 24.0 | topk - relu | -0.0114 | 0.0138 | a_lower |
| numerical | 3 | 24.0 | topk - jumprelu | -0.0066 | 0.0218 | mixed |
| numerical | 3 | 24.0 | jumprelu - relu | -0.0048 | 0.0239 | mixed |
| numerical | 3 | 32.0 | topk - relu | 0.0092 | 0.0035 | a_higher |
| numerical | 3 | 32.0 | topk - jumprelu | -0.0009 | 0.0082 | mixed |
| numerical | 3 | 32.0 | jumprelu - relu | 0.0102 | 0.0049 | a_higher |
| numerical | 4 | 8.0 | topk - relu | -0.0019 | 0.0033 | mixed |
| numerical | 4 | 8.0 | topk - jumprelu | -0.0107 | 0.0054 | a_lower |
| numerical | 4 | 8.0 | jumprelu - relu | 0.0088 | 0.004 | a_higher |
| numerical | 4 | 16.0 | topk - relu | 0.0009 | 0.0107 | mixed |
| numerical | 4 | 16.0 | topk - jumprelu | 0.005 | 0.007 | mixed |
| numerical | 4 | 16.0 | jumprelu - relu | -0.0041 | 0.0106 | mixed |
| numerical | 4 | 24.0 | topk - relu | 0.0271 | 0.0158 | a_higher |
| numerical | 4 | 24.0 | topk - jumprelu | 0.031 | 0.0068 | a_higher |
| numerical | 4 | 24.0 | jumprelu - relu | -0.0039 | 0.0136 | mixed |
| numerical | 4 | 32.0 | topk - relu | -0.0295 | 0.0325 | a_lower |
| numerical | 4 | 32.0 | topk - jumprelu | -0.0095 | 0.026 | mixed |
| numerical | 4 | 32.0 | jumprelu - relu | -0.02 | 0.0117 | a_lower |
| numerical | 5 | 8.0 | topk - relu | 0.0101 | 0.0391 | mixed |
| numerical | 5 | 8.0 | topk - jumprelu | 0.0037 | 0.0328 | mixed |
| numerical | 5 | 8.0 | jumprelu - relu | 0.0064 | 0.0076 | mixed |
| numerical | 5 | 16.0 | topk - relu | -0.0039 | 0.0314 | mixed |
| numerical | 5 | 16.0 | topk - jumprelu | 0.0063 | 0.0547 | mixed |
| numerical | 5 | 16.0 | jumprelu - relu | -0.0102 | 0.0347 | mixed |
| numerical | 5 | 24.0 | topk - relu | -0.0011 | 0.0136 | mixed |
| numerical | 5 | 24.0 | topk - jumprelu | -0.0026 | 0.0133 | mixed |
| numerical | 5 | 24.0 | jumprelu - relu | 0.0016 | 0.0117 | mixed |
| numerical | 5 | 32.0 | topk - relu | 0.024 | 0.0252 | a_higher |
| numerical | 5 | 32.0 | topk - jumprelu | 0.0007 | 0.0652 | mixed |
| numerical | 5 | 32.0 | jumprelu - relu | 0.0232 | 0.0424 | mixed |
| numerical | 5 | 48.0 | topk - relu | 0.0124 | 0.0381 | mixed |
| numerical | 5 | 48.0 | topk - jumprelu | 0.014 | 0.0187 | mixed |
| numerical | 5 | 48.0 | jumprelu - relu | -0.0016 | 0.0282 | mixed |
| numerical | 6 | 8.0 | topk - relu | 0.0171 | 0.0296 | mixed |
| numerical | 6 | 8.0 | topk - jumprelu | -0.0002 | 0.0173 | mixed |
| numerical | 6 | 8.0 | jumprelu - relu | 0.0173 | 0.0293 | mixed |
| numerical | 6 | 16.0 | topk - relu | 0.0205 | 0.0225 | mixed |
| numerical | 6 | 16.0 | topk - jumprelu | 0.0573 | 0.0359 | a_higher |
| numerical | 6 | 16.0 | jumprelu - relu | -0.0368 | 0.0478 | mixed |
| numerical | 6 | 24.0 | topk - relu | 0.0143 | 0.0468 | mixed |
| numerical | 6 | 24.0 | topk - jumprelu | -0.0036 | 0.0716 | mixed |
| numerical | 6 | 24.0 | jumprelu - relu | 0.0179 | 0.0523 | mixed |
| numerical | 6 | 32.0 | topk - relu | 0.0383 | 0.0311 | a_higher |
| numerical | 6 | 32.0 | topk - jumprelu | 0.0502 | 0.1002 | mixed |
| numerical | 6 | 32.0 | jumprelu - relu | -0.0119 | 0.0701 | mixed |
| numerical | 6 | 48.0 | topk - relu | 0.0322 | 0.0817 | mixed |
| numerical | 6 | 48.0 | topk - jumprelu | -0.0008 | 0.0494 | mixed |
| numerical | 6 | 48.0 | jumprelu - relu | 0.033 | 0.035 | a_higher |
| numerical | 7 | 8.0 | topk - relu | -0.0487 | 0.0714 | a_lower |
| numerical | 7 | 8.0 | topk - jumprelu | -0.0031 | 0.046 | mixed |
| numerical | 7 | 8.0 | jumprelu - relu | -0.0456 | 0.0286 | a_lower |
| numerical | 7 | 16.0 | topk - relu | 0.0434 | 0.061 | mixed |
| numerical | 7 | 16.0 | topk - jumprelu | -0.0149 | 0.0913 | mixed |
| numerical | 7 | 16.0 | jumprelu - relu | 0.0582 | 0.0493 | a_higher |
| numerical | 7 | 24.0 | topk - relu | -0.0248 | 0.0324 | a_lower |
| numerical | 7 | 24.0 | topk - jumprelu | 0.0027 | 0.0326 | mixed |
| numerical | 7 | 24.0 | jumprelu - relu | -0.0275 | 0.0475 | mixed |
| numerical | 7 | 32.0 | topk - relu | 0.0458 | 0.0177 | a_higher |
| numerical | 7 | 32.0 | topk - jumprelu | -0.029 | 0.0105 | a_lower |
| numerical | 7 | 32.0 | jumprelu - relu | 0.0748 | 0.0281 | a_higher |
| numerical | 7 | 48.0 | topk - relu | 0.0321 | 0.0858 | mixed |
| numerical | 7 | 48.0 | topk - jumprelu | 0.0367 | 0.1067 | mixed |
| numerical | 7 | 48.0 | jumprelu - relu | -0.0045 | 0.0248 | mixed |
| numerical | 7 | 64.0 | topk - relu | 0.0465 | 0.0651 | mixed |
| numerical | 7 | 64.0 | topk - jumprelu | 0.0474 | 0.0577 | mixed |
| numerical | 7 | 64.0 | jumprelu - relu | -0.0009 | 0.0121 | mixed |
| numerical | 8 | 8.0 | topk - relu | -0.0457 | 0.0333 | a_lower |
| numerical | 8 | 8.0 | topk - jumprelu | 0.0099 | 0.0253 | mixed |
| numerical | 8 | 8.0 | jumprelu - relu | -0.0557 | 0.0517 | a_lower |
| numerical | 8 | 16.0 | topk - relu | 0.0012 | 0.0431 | mixed |
| numerical | 8 | 16.0 | topk - jumprelu | 0.0009 | 0.0141 | mixed |
| numerical | 8 | 16.0 | jumprelu - relu | 0.0003 | 0.0514 | mixed |
| numerical | 8 | 24.0 | topk - relu | -0.0339 | 0.0309 | mixed |
| numerical | 8 | 24.0 | topk - jumprelu | -0.0184 | 0.0274 | mixed |
| numerical | 8 | 24.0 | jumprelu - relu | -0.0155 | 0.0446 | mixed |
| numerical | 8 | 32.0 | topk - relu | 0.0421 | 0.0891 | mixed |
| numerical | 8 | 32.0 | topk - jumprelu | 0.0612 | 0.0771 | mixed |
| numerical | 8 | 32.0 | jumprelu - relu | -0.0191 | 0.0122 | a_lower |
| numerical | 8 | 48.0 | topk - relu | 0.007 | 0.0098 | a_higher |
| numerical | 8 | 48.0 | topk - jumprelu | 0.0329 | 0.0483 | mixed |
| numerical | 8 | 48.0 | jumprelu - relu | -0.026 | 0.0575 | mixed |
| numerical | 9 | 8.0 | topk - relu | 0.0093 | 0.0344 | mixed |
| numerical | 9 | 8.0 | topk - jumprelu | 0.0108 | 0.0227 | mixed |
| numerical | 9 | 8.0 | jumprelu - relu | -0.0014 | 0.0569 | mixed |
| numerical | 9 | 16.0 | topk - relu | 0.0435 | 0.0444 | mixed |
| numerical | 9 | 16.0 | topk - jumprelu | -0.0095 | 0.0615 | mixed |
| numerical | 9 | 16.0 | jumprelu - relu | 0.053 | 0.0759 | mixed |
| numerical | 9 | 24.0 | topk - relu | 0.0532 | 0.0969 | mixed |
| numerical | 9 | 24.0 | topk - jumprelu | 0.0298 | 0.0709 | mixed |
| numerical | 9 | 24.0 | jumprelu - relu | 0.0234 | 0.0678 | mixed |
| numerical | 9 | 32.0 | topk - relu | 0.0763 | 0.034 | a_higher |
| numerical | 9 | 32.0 | topk - jumprelu | 0.0362 | 0.0144 | a_higher |
| numerical | 9 | 32.0 | jumprelu - relu | 0.04 | 0.0265 | a_higher |
| numerical | 9 | 48.0 | topk - relu | -0.0659 | 0.0525 | a_lower |
| numerical | 9 | 48.0 | topk - jumprelu | -0.0154 | 0.0375 | mixed |
| numerical | 9 | 48.0 | jumprelu - relu | -0.0506 | 0.037 | a_lower |
| numerical | 9 | 64.0 | topk - relu | 0.0365 | 0.0653 | mixed |
| numerical | 9 | 64.0 | topk - jumprelu | 0.0528 | 0.0328 | a_higher |
| numerical | 9 | 64.0 | jumprelu - relu | -0.0162 | 0.045 | mixed |
| numerical | 10 | 8.0 | topk - relu | -0.025 | 0.0071 | a_lower |
| numerical | 10 | 8.0 | topk - jumprelu | -0.01 | 0.0428 | mixed |
| numerical | 10 | 8.0 | jumprelu - relu | -0.015 | 0.0474 | mixed |
| numerical | 10 | 16.0 | topk - relu | 0.0814 | 0.0491 | a_higher |
| numerical | 10 | 16.0 | topk - jumprelu | 0.0441 | 0.0665 | mixed |
| numerical | 10 | 16.0 | jumprelu - relu | 0.0373 | 0.0174 | a_higher |
| numerical | 10 | 24.0 | topk - relu | 0.0318 | 0.0512 | mixed |
| numerical | 10 | 24.0 | topk - jumprelu | 0.0352 | 0.0303 | a_higher |
| numerical | 10 | 24.0 | jumprelu - relu | -0.0034 | 0.0815 | mixed |
| numerical | 10 | 32.0 | topk - relu | 0.0216 | 0.0344 | mixed |
| numerical | 10 | 32.0 | topk - jumprelu | 0.0147 | 0.0358 | mixed |
| numerical | 10 | 32.0 | jumprelu - relu | 0.0069 | 0.0674 | mixed |
| numerical | 10 | 48.0 | topk - relu | -0.0183 | 0.041 | mixed |
| numerical | 10 | 48.0 | topk - jumprelu | 0.0088 | 0.1024 | mixed |
| numerical | 10 | 48.0 | jumprelu - relu | -0.0271 | 0.0683 | mixed |
| numerical | 10 | 64.0 | topk - relu | 0.0193 | 0.0301 | mixed |
| numerical | 10 | 64.0 | topk - jumprelu | 0.0387 | 0.0333 | a_higher |
| numerical | 10 | 64.0 | jumprelu - relu | -0.0193 | 0.0427 | mixed |
| numerical | 11 | 8.0 | topk - relu | -0.0852 | 0.0447 | a_lower |
| numerical | 11 | 8.0 | topk - jumprelu | -0.0049 | 0.0189 | mixed |
| numerical | 11 | 8.0 | jumprelu - relu | -0.0803 | 0.0401 | a_lower |
| numerical | 11 | 16.0 | topk - relu | -0.0204 | 0.0207 | a_lower |
| numerical | 11 | 16.0 | topk - jumprelu | -0.0278 | 0.0228 | a_lower |
| numerical | 11 | 16.0 | jumprelu - relu | 0.0074 | 0.0116 | mixed |
| numerical | 11 | 24.0 | topk - relu | -0.0031 | 0.0496 | mixed |
| numerical | 11 | 24.0 | topk - jumprelu | -0.024 | 0.0484 | mixed |
| numerical | 11 | 24.0 | jumprelu - relu | 0.021 | 0.0166 | a_higher |
| numerical | 11 | 32.0 | topk - relu | 0.0356 | 0.0205 | a_higher |
| numerical | 11 | 32.0 | topk - jumprelu | -0.0132 | 0.0109 | a_lower |
| numerical | 11 | 32.0 | jumprelu - relu | 0.0488 | 0.0299 | a_higher |
| numerical | 11 | 48.0 | topk - relu | 0.0586 | 0.0474 | a_higher |
| numerical | 11 | 48.0 | topk - jumprelu | 0.0346 | 0.0676 | mixed |
| numerical | 11 | 48.0 | jumprelu - relu | 0.024 | 0.0212 | a_higher |
| numerical | 11 | 64.0 | topk - relu | 0.0668 | 0.0202 | a_higher |
| numerical | 11 | 64.0 | topk - jumprelu | -0.0059 | 0.0347 | mixed |
| numerical | 11 | 64.0 | jumprelu - relu | 0.0727 | 0.0362 | a_higher |

## Matched-L0 comparison (repe, logistic gap, informative layers)

| domain | layer | m | comparison | mean_diff | sd_diff | all_seeds_agree |
|---|---|---|---|---|---|---|
| honesty | 0 | 8.0 | topk - relu | 0.0245 | 0.0521 | mixed |
| honesty | 0 | 8.0 | topk - jumprelu | 0.022 | 0.0197 | mixed |
| honesty | 0 | 8.0 | jumprelu - relu | 0.0025 | 0.0351 | mixed |
| honesty | 1 | 8.0 | topk - relu | 0.036 | 0.0363 | a_higher |
| honesty | 1 | 8.0 | topk - jumprelu | -0.1003 | 0.0199 | a_lower |
| honesty | 1 | 8.0 | jumprelu - relu | 0.1363 | 0.0203 | a_higher |
| honesty | 1 | 16.0 | topk - relu | 0.0064 | 0.0294 | mixed |
| honesty | 1 | 16.0 | topk - jumprelu | -0.0549 | 0.0636 | mixed |
| honesty | 1 | 16.0 | jumprelu - relu | 0.0613 | 0.0424 | a_higher |
| honesty | 1 | 24.0 | topk - relu | 0.0494 | 0.0411 | a_higher |
| honesty | 1 | 24.0 | topk - jumprelu | -0.0006 | 0.0669 | mixed |
| honesty | 1 | 24.0 | jumprelu - relu | 0.05 | 0.052 | a_higher |
| honesty | 1 | 32.0 | topk - relu | 0.0737 | 0.0319 | a_higher |
| honesty | 1 | 32.0 | topk - jumprelu | -0.015 | 0.0261 | mixed |
| honesty | 1 | 32.0 | jumprelu - relu | 0.0888 | 0.0107 | a_higher |
| honesty | 1 | 48.0 | topk - relu | 0.0491 | 0.051 | a_higher |
| honesty | 1 | 48.0 | topk - jumprelu | -0.0365 | 0.0497 | mixed |
| honesty | 1 | 48.0 | jumprelu - relu | 0.0856 | 0.0023 | a_higher |
| honesty | 2 | 8.0 | topk - relu | 0.0304 | 0.0124 | a_higher |
| honesty | 2 | 8.0 | topk - jumprelu | -0.0196 | 0.0417 | mixed |
| honesty | 2 | 8.0 | jumprelu - relu | 0.05 | 0.0458 | mixed |
| honesty | 2 | 16.0 | topk - relu | -0.009 | 0.0303 | mixed |
| honesty | 2 | 16.0 | topk - jumprelu | -0.0937 | 0.0208 | a_lower |
| honesty | 2 | 16.0 | jumprelu - relu | 0.0848 | 0.027 | a_higher |
| honesty | 2 | 24.0 | topk - relu | -0.0244 | 0.0106 | a_lower |
| honesty | 2 | 24.0 | topk - jumprelu | -0.1204 | 0.0273 | a_lower |
| honesty | 2 | 24.0 | jumprelu - relu | 0.096 | 0.017 | a_higher |
| honesty | 2 | 32.0 | topk - relu | -0.0307 | 0.0221 | a_lower |
| honesty | 2 | 32.0 | topk - jumprelu | -0.1346 | 0.0551 | a_lower |
| honesty | 2 | 32.0 | jumprelu - relu | 0.1039 | 0.0334 | a_higher |
| honesty | 2 | 48.0 | topk - relu | -0.0482 | 0.0352 | a_lower |
| honesty | 2 | 48.0 | topk - jumprelu | -0.1526 | 0.0596 | a_lower |
| honesty | 2 | 48.0 | jumprelu - relu | 0.1043 | 0.0248 | a_higher |
| honesty | 2 | 64.0 | topk - relu | -0.0442 | 0.0309 | a_lower |
| honesty | 2 | 64.0 | topk - jumprelu | -0.1404 | 0.0517 | a_lower |
| honesty | 2 | 64.0 | jumprelu - relu | 0.0961 | 0.0213 | a_higher |
| honesty | 3 | 8.0 | topk - relu | 0.0047 | 0.0309 | mixed |
| honesty | 3 | 8.0 | topk - jumprelu | -0.0185 | 0.0053 | a_lower |
| honesty | 3 | 8.0 | jumprelu - relu | 0.0233 | 0.0256 | mixed |
| honesty | 3 | 16.0 | topk - relu | 0.0141 | 0.0098 | a_higher |
| honesty | 3 | 16.0 | topk - jumprelu | -0.0776 | 0.0082 | a_lower |
| honesty | 3 | 16.0 | jumprelu - relu | 0.0917 | 0.0166 | a_higher |
| honesty | 3 | 24.0 | topk - relu | 0.0345 | 0.0344 | a_higher |
| honesty | 3 | 24.0 | topk - jumprelu | -0.1002 | 0.0456 | a_lower |
| honesty | 3 | 24.0 | jumprelu - relu | 0.1347 | 0.0123 | a_higher |
| honesty | 3 | 32.0 | topk - relu | 0.0217 | 0.0436 | mixed |
| honesty | 3 | 32.0 | topk - jumprelu | -0.12 | 0.0523 | a_lower |
| honesty | 3 | 32.0 | jumprelu - relu | 0.1416 | 0.0092 | a_higher |
| honesty | 3 | 48.0 | topk - relu | -0.0361 | 0.0487 | mixed |
| honesty | 3 | 48.0 | topk - jumprelu | -0.1881 | 0.0712 | a_lower |
| honesty | 3 | 48.0 | jumprelu - relu | 0.152 | 0.0226 | a_higher |
| honesty | 3 | 64.0 | topk - relu | -0.0413 | 0.0241 | a_lower |
| honesty | 3 | 64.0 | topk - jumprelu | -0.1854 | 0.0386 | a_lower |
| honesty | 3 | 64.0 | jumprelu - relu | 0.1441 | 0.0178 | a_higher |
| honesty | 4 | 8.0 | topk - relu | -0.0304 | 0.0506 | mixed |
| honesty | 4 | 8.0 | topk - jumprelu | -0.0063 | 0.0091 | mixed |
| honesty | 4 | 8.0 | jumprelu - relu | -0.0241 | 0.0489 | mixed |
| honesty | 4 | 16.0 | topk - relu | 0.0164 | 0.0201 | mixed |
| honesty | 4 | 16.0 | topk - jumprelu | -0.0693 | 0.0199 | a_lower |
| honesty | 4 | 16.0 | jumprelu - relu | 0.0857 | 0.0233 | a_higher |
| honesty | 4 | 24.0 | topk - relu | 0.0357 | 0.0264 | a_higher |
| honesty | 4 | 24.0 | topk - jumprelu | -0.0829 | 0.034 | a_lower |
| honesty | 4 | 24.0 | jumprelu - relu | 0.1186 | 0.0122 | a_higher |
| honesty | 4 | 32.0 | topk - relu | 0.0149 | 0.0522 | mixed |
| honesty | 4 | 32.0 | topk - jumprelu | -0.117 | 0.0603 | a_lower |
| honesty | 4 | 32.0 | jumprelu - relu | 0.1319 | 0.03 | a_higher |
| honesty | 4 | 48.0 | topk - relu | 0.0176 | 0.0252 | a_higher |
| honesty | 4 | 48.0 | topk - jumprelu | -0.1303 | 0.0426 | a_lower |
| honesty | 4 | 48.0 | jumprelu - relu | 0.1479 | 0.0209 | a_higher |
| honesty | 4 | 64.0 | topk - relu | -0.0132 | 0.0107 | a_lower |
| honesty | 4 | 64.0 | topk - jumprelu | -0.1481 | 0.039 | a_lower |
| honesty | 4 | 64.0 | jumprelu - relu | 0.135 | 0.0299 | a_higher |
| honesty | 4 | 96.0 | topk - relu | -0.0495 | 0.0324 | a_lower |
| honesty | 4 | 96.0 | topk - jumprelu | -0.1387 | 0.0185 | a_lower |
| honesty | 4 | 96.0 | jumprelu - relu | 0.0893 | 0.044 | a_higher |
| honesty | 5 | 8.0 | topk - relu | -0.1271 | 0.0259 | a_lower |
| honesty | 5 | 8.0 | topk - jumprelu | -0.1093 | 0.0943 | a_lower |
| honesty | 5 | 8.0 | jumprelu - relu | -0.0178 | 0.0686 | mixed |
| honesty | 5 | 16.0 | topk - relu | -0.0834 | 0.0303 | a_lower |
| honesty | 5 | 16.0 | topk - jumprelu | -0.0726 | 0.0312 | a_lower |
| honesty | 5 | 16.0 | jumprelu - relu | -0.0108 | 0.0458 | mixed |
| honesty | 5 | 24.0 | topk - relu | -0.0581 | 0.0543 | mixed |
| honesty | 5 | 24.0 | topk - jumprelu | -0.053 | 0.0134 | a_lower |
| honesty | 5 | 24.0 | jumprelu - relu | -0.0051 | 0.0591 | mixed |
| honesty | 5 | 32.0 | topk - relu | -0.0472 | 0.0441 | mixed |
| honesty | 5 | 32.0 | topk - jumprelu | -0.0549 | 0.0252 | a_lower |
| honesty | 5 | 32.0 | jumprelu - relu | 0.0078 | 0.0463 | mixed |
| honesty | 5 | 48.0 | topk - relu | -0.0533 | 0.0481 | a_lower |
| honesty | 5 | 48.0 | topk - jumprelu | -0.0687 | 0.0451 | a_lower |
| honesty | 5 | 48.0 | jumprelu - relu | 0.0155 | 0.0685 | mixed |
| honesty | 5 | 64.0 | topk - relu | -0.0634 | 0.0397 | a_lower |
| honesty | 5 | 64.0 | topk - jumprelu | -0.066 | 0.0612 | mixed |
| honesty | 5 | 64.0 | jumprelu - relu | 0.0027 | 0.0696 | mixed |
| honesty | 5 | 96.0 | topk - relu | -0.103 | 0.0234 | a_lower |
| honesty | 5 | 96.0 | topk - jumprelu | -0.0869 | 0.0609 | a_lower |
| honesty | 5 | 96.0 | jumprelu - relu | -0.0161 | 0.0658 | mixed |
| honesty | 6 | 8.0 | topk - relu | -0.0672 | 0.0516 | a_lower |
| honesty | 6 | 8.0 | topk - jumprelu | 0.0075 | 0.0511 | mixed |
| honesty | 6 | 8.0 | jumprelu - relu | -0.0747 | 0.0117 | a_lower |
| honesty | 6 | 16.0 | topk - relu | -0.0693 | 0.0224 | a_lower |
| honesty | 6 | 16.0 | topk - jumprelu | -0.0244 | 0.0177 | a_lower |
| honesty | 6 | 16.0 | jumprelu - relu | -0.045 | 0.0185 | a_lower |
| honesty | 6 | 24.0 | topk - relu | -0.0666 | 0.0113 | a_lower |
| honesty | 6 | 24.0 | topk - jumprelu | -0.0577 | 0.0162 | a_lower |
| honesty | 6 | 24.0 | jumprelu - relu | -0.0089 | 0.0265 | mixed |
| honesty | 6 | 32.0 | topk - relu | -0.0651 | 0.0214 | a_lower |
| honesty | 6 | 32.0 | topk - jumprelu | -0.0596 | 0.0046 | a_lower |
| honesty | 6 | 32.0 | jumprelu - relu | -0.0055 | 0.0256 | mixed |
| honesty | 6 | 48.0 | topk - relu | -0.0618 | 0.0114 | a_lower |
| honesty | 6 | 48.0 | topk - jumprelu | -0.0719 | 0.0285 | a_lower |
| honesty | 6 | 48.0 | jumprelu - relu | 0.0101 | 0.0198 | mixed |
| honesty | 6 | 64.0 | topk - relu | -0.0884 | 0.0056 | a_lower |
| honesty | 6 | 64.0 | topk - jumprelu | -0.0915 | 0.0176 | a_lower |
| honesty | 6 | 64.0 | jumprelu - relu | 0.0031 | 0.0173 | mixed |
| honesty | 6 | 96.0 | topk - relu | -0.1376 | 0.0273 | a_lower |
| honesty | 6 | 96.0 | topk - jumprelu | -0.0828 | 0.0211 | a_lower |
| honesty | 6 | 96.0 | jumprelu - relu | -0.0547 | 0.0483 | a_lower |
| honesty | 7 | 8.0 | topk - relu | -0.0705 | 0.0205 | a_lower |
| honesty | 7 | 8.0 | topk - jumprelu | -0.0026 | 0.0372 | mixed |
| honesty | 7 | 8.0 | jumprelu - relu | -0.0679 | 0.0184 | a_lower |
| honesty | 7 | 16.0 | topk - relu | -0.0931 | 0.0476 | a_lower |
| honesty | 7 | 16.0 | topk - jumprelu | -0.0637 | 0.0373 | a_lower |
| honesty | 7 | 16.0 | jumprelu - relu | -0.0294 | 0.0102 | a_lower |
| honesty | 7 | 24.0 | topk - relu | -0.1389 | 0.0523 | a_lower |
| honesty | 7 | 24.0 | topk - jumprelu | -0.0953 | 0.0441 | a_lower |
| honesty | 7 | 24.0 | jumprelu - relu | -0.0436 | 0.0209 | a_lower |
| honesty | 7 | 32.0 | topk - relu | -0.1048 | 0.054 | a_lower |
| honesty | 7 | 32.0 | topk - jumprelu | -0.0905 | 0.0532 | a_lower |
| honesty | 7 | 32.0 | jumprelu - relu | -0.0143 | 0.0172 | mixed |
| honesty | 7 | 48.0 | topk - relu | -0.1521 | 0.0529 | a_lower |
| honesty | 7 | 48.0 | topk - jumprelu | -0.0953 | 0.0705 | a_lower |
| honesty | 7 | 48.0 | jumprelu - relu | -0.0567 | 0.0283 | a_lower |
| honesty | 7 | 64.0 | topk - relu | -0.1658 | 0.052 | a_lower |
| honesty | 7 | 64.0 | topk - jumprelu | -0.0978 | 0.0735 | a_lower |
| honesty | 7 | 64.0 | jumprelu - relu | -0.068 | 0.0386 | a_lower |
| honesty | 7 | 96.0 | topk - relu | -0.1915 | 0.0179 | a_lower |
| honesty | 7 | 96.0 | topk - jumprelu | -0.1099 | 0.0767 | a_lower |
| honesty | 7 | 96.0 | jumprelu - relu | -0.0816 | 0.0631 | a_lower |
| honesty | 7 | 128.0 | topk - relu | -0.2049 | 0.0073 | a_lower |
| honesty | 7 | 128.0 | topk - jumprelu | -0.0834 | 0.0463 | a_lower |
| honesty | 7 | 128.0 | jumprelu - relu | -0.1215 | 0.0423 | a_lower |
| honesty | 8 | 8.0 | topk - relu | -0.1089 | 0.0813 | a_lower |
| honesty | 8 | 8.0 | topk - jumprelu | -0.1411 | 0.0771 | a_lower |
| honesty | 8 | 8.0 | jumprelu - relu | 0.0322 | 0.0432 | mixed |
| honesty | 8 | 16.0 | topk - relu | -0.1095 | 0.0428 | a_lower |
| honesty | 8 | 16.0 | topk - jumprelu | -0.0444 | 0.0356 | a_lower |
| honesty | 8 | 16.0 | jumprelu - relu | -0.0651 | 0.0149 | a_lower |
| honesty | 8 | 24.0 | topk - relu | -0.105 | 0.0498 | a_lower |
| honesty | 8 | 24.0 | topk - jumprelu | -0.0846 | 0.0483 | a_lower |
| honesty | 8 | 24.0 | jumprelu - relu | -0.0204 | 0.0162 | a_lower |
| honesty | 8 | 32.0 | topk - relu | -0.1129 | 0.0378 | a_lower |
| honesty | 8 | 32.0 | topk - jumprelu | -0.0844 | 0.021 | a_lower |
| honesty | 8 | 32.0 | jumprelu - relu | -0.0284 | 0.0169 | a_lower |
| honesty | 8 | 48.0 | topk - relu | -0.1371 | 0.04 | a_lower |
| honesty | 8 | 48.0 | topk - jumprelu | -0.114 | 0.0179 | a_lower |
| honesty | 8 | 48.0 | jumprelu - relu | -0.0232 | 0.0261 | mixed |
| honesty | 8 | 64.0 | topk - relu | -0.167 | 0.0313 | a_lower |
| honesty | 8 | 64.0 | topk - jumprelu | -0.1388 | 0.0092 | a_lower |
| honesty | 8 | 64.0 | jumprelu - relu | -0.0283 | 0.0228 | a_lower |
| honesty | 8 | 96.0 | topk - relu | -0.1918 | 0.0362 | a_lower |
| honesty | 8 | 96.0 | topk - jumprelu | -0.1167 | 0.026 | a_lower |
| honesty | 8 | 96.0 | jumprelu - relu | -0.0751 | 0.0334 | a_lower |
| honesty | 8 | 128.0 | topk - relu | -0.2095 | 0.0258 | a_lower |
| honesty | 8 | 128.0 | topk - jumprelu | -0.0783 | 0.0666 | a_lower |
| honesty | 8 | 128.0 | jumprelu - relu | -0.1312 | 0.0527 | a_lower |
| honesty | 9 | 8.0 | topk - relu | -0.0708 | 0.0768 | mixed |
| honesty | 9 | 8.0 | topk - jumprelu | -0.146 | 0.0837 | a_lower |
| honesty | 9 | 8.0 | jumprelu - relu | 0.0753 | 0.007 | a_higher |
| honesty | 9 | 16.0 | topk - relu | -0.1149 | 0.0844 | a_lower |
| honesty | 9 | 16.0 | topk - jumprelu | -0.1136 | 0.1021 | mixed |
| honesty | 9 | 16.0 | jumprelu - relu | -0.0014 | 0.062 | mixed |
| honesty | 9 | 24.0 | topk - relu | -0.1105 | 0.0824 | a_lower |
| honesty | 9 | 24.0 | topk - jumprelu | -0.0711 | 0.0937 | mixed |
| honesty | 9 | 24.0 | jumprelu - relu | -0.0393 | 0.0409 | mixed |
| honesty | 9 | 32.0 | topk - relu | -0.1158 | 0.0731 | a_lower |
| honesty | 9 | 32.0 | topk - jumprelu | -0.0622 | 0.0784 | mixed |
| honesty | 9 | 32.0 | jumprelu - relu | -0.0536 | 0.0167 | a_lower |
| honesty | 9 | 48.0 | topk - relu | -0.1415 | 0.0676 | a_lower |
| honesty | 9 | 48.0 | topk - jumprelu | -0.0773 | 0.0535 | a_lower |
| honesty | 9 | 48.0 | jumprelu - relu | -0.0642 | 0.0174 | a_lower |
| honesty | 9 | 64.0 | topk - relu | -0.1517 | 0.0623 | a_lower |
| honesty | 9 | 64.0 | topk - jumprelu | -0.0948 | 0.0677 | a_lower |
| honesty | 9 | 64.0 | jumprelu - relu | -0.0569 | 0.0338 | a_lower |
| honesty | 9 | 96.0 | topk - relu | -0.1811 | 0.0684 | a_lower |
| honesty | 9 | 96.0 | topk - jumprelu | -0.0742 | 0.0576 | a_lower |
| honesty | 9 | 96.0 | jumprelu - relu | -0.1069 | 0.02 | a_lower |
| honesty | 9 | 128.0 | topk - relu | -0.1902 | 0.072 | a_lower |
| honesty | 9 | 128.0 | topk - jumprelu | -0.0492 | 0.0523 | mixed |
| honesty | 9 | 128.0 | jumprelu - relu | -0.1409 | 0.0197 | a_lower |
| honesty | 10 | 8.0 | topk - relu | 0.032 | 0.0116 | a_higher |
| honesty | 10 | 8.0 | topk - jumprelu | -0.0031 | 0.049 | mixed |
| honesty | 10 | 8.0 | jumprelu - relu | 0.0351 | 0.0452 | mixed |
| honesty | 10 | 16.0 | topk - relu | -0.0526 | 0.0414 | a_lower |
| honesty | 10 | 16.0 | topk - jumprelu | -0.0651 | 0.0283 | a_lower |
| honesty | 10 | 16.0 | jumprelu - relu | 0.0125 | 0.0281 | mixed |
| honesty | 10 | 24.0 | topk - relu | -0.0788 | 0.0373 | a_lower |
| honesty | 10 | 24.0 | topk - jumprelu | -0.0461 | 0.0257 | a_lower |
| honesty | 10 | 24.0 | jumprelu - relu | -0.0328 | 0.0138 | a_lower |
| honesty | 10 | 32.0 | topk - relu | -0.0918 | 0.0333 | a_lower |
| honesty | 10 | 32.0 | topk - jumprelu | -0.047 | 0.038 | a_lower |
| honesty | 10 | 32.0 | jumprelu - relu | -0.0448 | 0.0074 | a_lower |
| honesty | 10 | 48.0 | topk - relu | -0.1206 | 0.0138 | a_lower |
| honesty | 10 | 48.0 | topk - jumprelu | -0.0594 | 0.0129 | a_lower |
| honesty | 10 | 48.0 | jumprelu - relu | -0.0612 | 0.0065 | a_lower |
| honesty | 10 | 64.0 | topk - relu | -0.1318 | 0.0361 | a_lower |
| honesty | 10 | 64.0 | topk - jumprelu | -0.0549 | 0.0249 | a_lower |
| honesty | 10 | 64.0 | jumprelu - relu | -0.0769 | 0.0123 | a_lower |
| honesty | 10 | 96.0 | topk - relu | -0.1526 | 0.0487 | a_lower |
| honesty | 10 | 96.0 | topk - jumprelu | -0.0506 | 0.0437 | a_lower |
| honesty | 10 | 96.0 | jumprelu - relu | -0.1019 | 0.0129 | a_lower |
| honesty | 10 | 128.0 | topk - relu | -0.1721 | 0.0596 | a_lower |
| honesty | 10 | 128.0 | topk - jumprelu | -0.0575 | 0.0573 | a_lower |
| honesty | 10 | 128.0 | jumprelu - relu | -0.1146 | 0.0242 | a_lower |
| honesty | 11 | 8.0 | topk - relu | 0.0409 | 0.068 | mixed |
| honesty | 11 | 8.0 | topk - jumprelu | -0.0075 | 0.0588 | mixed |
| honesty | 11 | 8.0 | jumprelu - relu | 0.0484 | 0.0144 | a_higher |
| honesty | 11 | 16.0 | topk - relu | -0.0503 | 0.0266 | a_lower |
| honesty | 11 | 16.0 | topk - jumprelu | -0.0779 | 0.0357 | a_lower |
| honesty | 11 | 16.0 | jumprelu - relu | 0.0276 | 0.0092 | a_higher |
| honesty | 11 | 24.0 | topk - relu | -0.1033 | 0.0483 | a_lower |
| honesty | 11 | 24.0 | topk - jumprelu | -0.0995 | 0.0456 | a_lower |
| honesty | 11 | 24.0 | jumprelu - relu | -0.0037 | 0.0147 | mixed |
| honesty | 11 | 32.0 | topk - relu | -0.1356 | 0.0548 | a_lower |
| honesty | 11 | 32.0 | topk - jumprelu | -0.1202 | 0.0694 | a_lower |
| honesty | 11 | 32.0 | jumprelu - relu | -0.0154 | 0.0192 | mixed |
| honesty | 11 | 48.0 | topk - relu | -0.1541 | 0.044 | a_lower |
| honesty | 11 | 48.0 | topk - jumprelu | -0.1163 | 0.046 | a_lower |
| honesty | 11 | 48.0 | jumprelu - relu | -0.0377 | 0.0388 | mixed |
| honesty | 11 | 64.0 | topk - relu | -0.1704 | 0.0537 | a_lower |
| honesty | 11 | 64.0 | topk - jumprelu | -0.1059 | 0.0563 | a_lower |
| honesty | 11 | 64.0 | jumprelu - relu | -0.0645 | 0.0465 | a_lower |
| honesty | 11 | 96.0 | topk - relu | -0.1854 | 0.0402 | a_lower |
| honesty | 11 | 96.0 | topk - jumprelu | -0.0955 | 0.0651 | a_lower |
| honesty | 11 | 96.0 | jumprelu - relu | -0.0899 | 0.0725 | a_lower |
| honesty | 11 | 128.0 | topk - relu | -0.1907 | 0.0546 | a_lower |
| honesty | 11 | 128.0 | topk - jumprelu | -0.0727 | 0.0738 | mixed |
| honesty | 11 | 128.0 | jumprelu - relu | -0.118 | 0.0906 | a_lower |

## Geometry of Truth causal steering (cities, test pairs)

| layer | part | intervention | sae_architecture | sae_seed | sign | detail | effect | ci95_low | ci95_high | p_upper | p_two_sided | p_bh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | A | unembedding_contrast |  |  | 0 | lambda=0.01 | 0.0 | 0.0 | 0.0 | 0.0017 | 0.0033 | 0.0017 |
| 4 | A | unembedding_contrast |  |  | 0 | lambda=0.02 | 0.0001 | 0.0 | 0.0001 | 0.0017 | 0.0033 | 0.0017 |
| 4 | A | unembedding_contrast |  |  | 0 | lambda=0.05 | 0.0001 | 0.0001 | 0.0001 | 0.0017 | 0.0033 | 0.0017 |
| 4 | A | unembedding_contrast |  |  | 0 | lambda=0.1 | 0.0003 | 0.0002 | 0.0003 | 0.0017 | 0.0033 | 0.0017 |
| 4 | A | unembedding_contrast |  |  | 0 | lambda=0.25 | 0.0006 | 0.0006 | 0.0007 | 0.0017 | 0.0033 | 0.0017 |
| 4 | A | unembedding_contrast |  |  | 0 | lambda=0.5 | 0.0013 | 0.0012 | 0.0014 | 0.0017 | 0.0033 | 0.0017 |
| 4 | A | unembedding_contrast |  |  | 0 | lambda=1.0 | 0.0026 | 0.0025 | 0.0027 | 0.0017 | 0.0033 | 0.0017 |
| 4 | C | mass_mean_direction |  |  | 1 |  | -0.0006 | -0.0007 | -0.0005 | 0.9334 | 0.1364 | 0.9468 |
| 4 | C | mass_mean_direction |  |  | -1 |  | 0.0006 | 0.0005 | 0.0007 | 0.0699 | 0.1398 |  |
| 4 | C | logistic_probe_direction |  |  | 1 |  | -0.0002 | -0.0003 | -0.0001 | 0.7155 | 0.5724 | 0.9468 |
| 4 | C | logistic_probe_direction |  |  | -1 |  | 0.0002 | 0.0001 | 0.0003 | 0.2995 | 0.599 |  |
| 4 | C | sae_reconstruction_direction | relu | 42.0 | 1 | cos_mass_mean=0.6232 | -0.0008 | -0.001 | -0.0006 | 0.9684 | 0.0666 | 0.6589 |
| 4 | C | sae_reconstruction_direction | relu | 42.0 | -1 | cos_mass_mean=0.6232 | 0.0008 | 0.0006 | 0.001 | 0.025 | 0.0499 |  |
| 4 | C | sae_top1_direction | relu | 42.0 | 1 | cos_mass_mean=0.2147 | -0.0 | -0.0002 | 0.0001 | 0.5225 | 0.9584 | 0.9983 |
| 4 | C | sae_top1_direction | relu | 42.0 | -1 | cos_mass_mean=0.2147 | 0.0 | -0.0001 | 0.0002 | 0.4559 | 0.9118 |  |
| 4 | C | sae_top10_direction | relu | 42.0 | 1 | cos_mass_mean=0.4931 | -0.0006 | -0.0007 | -0.0005 | 0.8885 | 0.2263 | 0.8927 |
| 4 | C | sae_top10_direction | relu | 42.0 | -1 | cos_mass_mean=0.4931 | 0.0006 | 0.0005 | 0.0007 | 0.1364 | 0.2729 |  |
| 4 | C | sae_reconstruction_direction | relu | 1042.0 | 1 | cos_mass_mean=0.6389 | -0.0003 | -0.0004 | -0.0002 | 0.7604 | 0.4825 | 0.9749 |
| 4 | C | sae_reconstruction_direction | relu | 1042.0 | -1 | cos_mass_mean=0.6389 | 0.0003 | 0.0002 | 0.0004 | 0.2662 | 0.5324 |  |
| 4 | C | sae_top1_direction | relu | 1042.0 | 1 | cos_mass_mean=0.2326 | -0.0001 | -0.0003 | -0.0 | 0.6639 | 0.6755 | 0.9749 |
| 4 | C | sae_top1_direction | relu | 1042.0 | -1 | cos_mass_mean=0.2326 | 0.0001 | 0.0 | 0.0003 | 0.3627 | 0.7255 |  |
| 4 | C | sae_top10_direction | relu | 1042.0 | 1 | cos_mass_mean=0.5689 | 0.0001 | -0.0 | 0.0002 | 0.4526 | 0.9052 | 0.9848 |
| 4 | C | sae_top10_direction | relu | 1042.0 | -1 | cos_mass_mean=0.5689 | -0.0001 | -0.0002 | 0.0 | 0.594 | 0.8153 |  |
| 4 | C | sae_reconstruction_direction | relu | 2042.0 | 1 | cos_mass_mean=0.6387 | -0.0006 | -0.0008 | -0.0005 | 0.9418 | 0.1198 | 0.8034 |
| 4 | C | sae_reconstruction_direction | relu | 2042.0 | -1 | cos_mass_mean=0.6387 | 0.0006 | 0.0005 | 0.0008 | 0.0566 | 0.1131 |  |
| 4 | C | sae_top1_direction | relu | 2042.0 | 1 | cos_mass_mean=0.1458 | -0.0003 | -0.0004 | -0.0002 | 0.7504 | 0.5025 | 0.9749 |
| 4 | C | sae_top1_direction | relu | 2042.0 | -1 | cos_mass_mean=0.1458 | 0.0003 | 0.0002 | 0.0004 | 0.2346 | 0.4692 |  |
| 4 | C | sae_top10_direction | relu | 2042.0 | 1 | cos_mass_mean=0.4944 | -0.0001 | -0.0002 | -0.0 | 0.6156 | 0.772 | 0.9779 |
| 4 | C | sae_top10_direction | relu | 2042.0 | -1 | cos_mass_mean=0.4944 | 0.0001 | 0.0 | 0.0002 | 0.4027 | 0.8053 |  |
| 4 | C | sae_reconstruction_direction | topk | 42.0 | 1 | cos_mass_mean=0.7264 | -0.0008 | -0.0009 | -0.0006 | 0.9651 | 0.0732 | 0.6589 |
| 4 | C | sae_reconstruction_direction | topk | 42.0 | -1 | cos_mass_mean=0.7264 | 0.0008 | 0.0006 | 0.0009 | 0.03 | 0.0599 |  |
| 4 | C | sae_top1_direction | topk | 42.0 | 1 | cos_mass_mean=0.2517 | -0.0002 | -0.0003 | -0.0 | 0.6473 | 0.7088 | 0.9749 |
| 4 | C | sae_top1_direction | topk | 42.0 | -1 | cos_mass_mean=0.2517 | 0.0002 | 0.0 | 0.0003 | 0.3195 | 0.6389 |  |
| 4 | C | sae_top10_direction | topk | 42.0 | 1 | cos_mass_mean=0.4650 | -0.0005 | -0.0006 | -0.0003 | 0.8153 | 0.3727 | 0.9749 |
| 4 | C | sae_top10_direction | topk | 42.0 | -1 | cos_mass_mean=0.4650 | 0.0005 | 0.0003 | 0.0006 | 0.1464 | 0.2928 |  |
| 4 | C | sae_reconstruction_direction | topk | 1042.0 | 1 | cos_mass_mean=0.7596 | -0.0009 | -0.0011 | -0.0008 | 0.9884 | 0.0266 | 0.599 |
| 4 | C | sae_reconstruction_direction | topk | 1042.0 | -1 | cos_mass_mean=0.7596 | 0.0009 | 0.0008 | 0.0011 | 0.0083 | 0.0166 |  |
| 4 | C | sae_top1_direction | topk | 1042.0 | 1 | cos_mass_mean=0.2643 | -0.0002 | -0.0003 | -0.0 | 0.6206 | 0.7621 | 0.9779 |
| 4 | C | sae_top1_direction | topk | 1042.0 | -1 | cos_mass_mean=0.2643 | 0.0002 | 0.0 | 0.0003 | 0.3494 | 0.6988 |  |
| 4 | C | sae_top10_direction | topk | 1042.0 | 1 | cos_mass_mean=0.5345 | -0.0004 | -0.0006 | -0.0003 | 0.8319 | 0.3394 | 0.94 |
| 4 | C | sae_top10_direction | topk | 1042.0 | -1 | cos_mass_mean=0.5345 | 0.0004 | 0.0003 | 0.0006 | 0.1581 | 0.3161 |  |
| 4 | C | sae_reconstruction_direction | topk | 2042.0 | 1 | cos_mass_mean=0.7618 | -0.0006 | -0.0008 | -0.0005 | 0.9384 | 0.1265 | 0.8034 |
| 4 | C | sae_reconstruction_direction | topk | 2042.0 | -1 | cos_mass_mean=0.7618 | 0.0006 | 0.0005 | 0.0008 | 0.0582 | 0.1165 |  |
| 4 | C | sae_top1_direction | topk | 2042.0 | 1 | cos_mass_mean=0.2675 | -0.0002 | -0.0004 | -0.0001 | 0.6739 | 0.6556 | 0.9749 |
| 4 | C | sae_top1_direction | topk | 2042.0 | -1 | cos_mass_mean=0.2675 | 0.0002 | 0.0001 | 0.0004 | 0.2745 | 0.5491 |  |
| 4 | C | sae_top10_direction | topk | 2042.0 | 1 | cos_mass_mean=0.4736 | -0.0011 | -0.0013 | -0.0009 | 0.985 | 0.0333 | 0.599 |
| 4 | C | sae_top10_direction | topk | 2042.0 | -1 | cos_mass_mean=0.4736 | 0.0011 | 0.0009 | 0.0013 | 0.0166 | 0.0333 |  |
| 4 | C | sae_reconstruction_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.5459 | -0.0007 | -0.0008 | -0.0006 | 0.9601 | 0.0832 | 0.6912 |
| 4 | C | sae_reconstruction_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.5459 | 0.0007 | 0.0006 | 0.0008 | 0.0383 | 0.0765 |  |
| 4 | C | sae_top1_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.2192 | -0.0005 | -0.0006 | -0.0004 | 0.8336 | 0.3361 | 0.94 |
| 4 | C | sae_top1_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.2192 | 0.0005 | 0.0004 | 0.0006 | 0.1498 | 0.2995 |  |
| 4 | C | sae_top10_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.4295 | -0.0005 | -0.0007 | -0.0004 | 0.8968 | 0.2097 | 0.8927 |
| 4 | C | sae_top10_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.4295 | 0.0005 | 0.0004 | 0.0007 | 0.1314 | 0.2629 |  |
| 4 | C | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.5833 | -0.0009 | -0.0011 | -0.0007 | 0.9867 | 0.03 | 0.599 |
| 4 | C | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.5833 | 0.0009 | 0.0007 | 0.0011 | 0.0083 | 0.0166 |  |
| 4 | C | sae_top1_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.2156 | -0.0013 | -0.0015 | -0.0011 | 0.9884 | 0.0266 | 0.599 |
| 4 | C | sae_top1_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.2156 | 0.0013 | 0.0011 | 0.0015 | 0.01 | 0.02 |  |
| 4 | C | sae_top10_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.3986 | -0.0013 | -0.0016 | -0.0011 | 0.9967 | 0.01 | 0.599 |
| 4 | C | sae_top10_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.3986 | 0.0013 | 0.0011 | 0.0016 | 0.005 | 0.01 |  |
| 4 | C | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.5712 | -0.0004 | -0.0005 | -0.0003 | 0.8353 | 0.3328 | 0.94 |
| 4 | C | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.5712 | 0.0004 | 0.0003 | 0.0005 | 0.1547 | 0.3095 |  |
| 4 | C | sae_top1_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.1980 | -0.0006 | -0.0007 | -0.0005 | 0.9018 | 0.1997 | 0.8927 |
| 4 | C | sae_top1_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.1980 | 0.0006 | 0.0005 | 0.0007 | 0.0948 | 0.1897 |  |
| 4 | C | sae_top10_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.5075 | -0.0004 | -0.0005 | -0.0003 | 0.8336 | 0.3361 | 0.94 |
| 4 | C | sae_top10_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.5075 | 0.0004 | 0.0003 | 0.0005 | 0.1997 | 0.3993 |  |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=0.01 | 0.0081 | 0.0078 | 0.0084 | 0.0017 | 0.0033 | 0.0017 |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=0.02 | 0.0162 | 0.0155 | 0.0169 | 0.0017 | 0.0033 | 0.0017 |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=0.05 | 0.0405 | 0.0388 | 0.0422 | 0.0017 | 0.0033 | 0.0017 |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=0.1 | 0.081 | 0.0776 | 0.0844 | 0.0017 | 0.0033 | 0.0017 |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=0.25 | 0.2021 | 0.1938 | 0.2106 | 0.0017 | 0.0033 | 0.0017 |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=0.5 | 0.4035 | 0.3868 | 0.4202 | 0.0017 | 0.0033 | 0.0017 |
| 6 | A | unembedding_contrast |  |  | 0 | lambda=1.0 | 0.8052 | 0.7721 | 0.8388 | 0.0017 | 0.0033 | 0.0017 |
| 6 | C | mass_mean_direction |  |  | 1 |  | 0.0047 | -0.014 | 0.0228 | 0.4592 | 0.9185 | 0.9468 |
| 6 | C | mass_mean_direction |  |  | -1 |  | -0.0166 | -0.0346 | 0.0021 | 0.5724 | 0.8586 |  |
| 6 | C | logistic_probe_direction |  |  | 1 |  | -0.1275 | -0.1552 | -0.1007 | 0.9468 | 0.1098 | 0.9468 |
| 6 | C | logistic_probe_direction |  |  | -1 |  | 0.1257 | 0.0987 | 0.1536 | 0.0632 | 0.1265 |  |
| 6 | C | sae_reconstruction_direction | relu | 42.0 | 1 | cos_mass_mean=0.6515 | -0.0329 | -0.0488 | -0.0175 | 0.6506 | 0.7022 | 0.9749 |
| 6 | C | sae_reconstruction_direction | relu | 42.0 | -1 | cos_mass_mean=0.6515 | 0.0482 | 0.0311 | 0.0657 | 0.2646 | 0.5291 |  |
| 6 | C | sae_top1_direction | relu | 42.0 | 1 | cos_mass_mean=0.2147 | 0.081 | 0.0595 | 0.1026 | 0.1597 | 0.3195 | 0.94 |
| 6 | C | sae_top1_direction | relu | 42.0 | -1 | cos_mass_mean=0.2147 | -0.0841 | -0.1053 | -0.0624 | 0.8336 | 0.3361 |  |
| 6 | C | sae_top10_direction | relu | 42.0 | 1 | cos_mass_mean=0.4753 | 0.0308 | 0.01 | 0.0507 | 0.2978 | 0.5957 | 0.9749 |
| 6 | C | sae_top10_direction | relu | 42.0 | -1 | cos_mass_mean=0.4753 | -0.0036 | -0.0235 | 0.0163 | 0.4459 | 0.8918 |  |
| 6 | C | sae_reconstruction_direction | relu | 1042.0 | 1 | cos_mass_mean=0.5549 | -0.0573 | -0.0748 | -0.0398 | 0.7704 | 0.4626 | 0.9749 |
| 6 | C | sae_reconstruction_direction | relu | 1042.0 | -1 | cos_mass_mean=0.5549 | 0.064 | 0.0463 | 0.0813 | 0.198 | 0.396 |  |
| 6 | C | sae_top1_direction | relu | 1042.0 | 1 | cos_mass_mean=0.1621 | -0.0769 | -0.0923 | -0.061 | 0.7754 | 0.4526 | 0.9749 |
| 6 | C | sae_top1_direction | relu | 1042.0 | -1 | cos_mass_mean=0.1621 | 0.0798 | 0.0631 | 0.0956 | 0.1714 | 0.3428 |  |
| 6 | C | sae_top10_direction | relu | 1042.0 | 1 | cos_mass_mean=0.3912 | -0.0838 | -0.1052 | -0.0624 | 0.7787 | 0.4459 | 0.9749 |
| 6 | C | sae_top10_direction | relu | 1042.0 | -1 | cos_mass_mean=0.3912 | 0.0831 | 0.0619 | 0.1037 | 0.1464 | 0.2928 |  |
| 6 | C | sae_reconstruction_direction | relu | 2042.0 | 1 | cos_mass_mean=0.5543 | -0.0831 | -0.1051 | -0.0601 | 0.8453 | 0.3128 | 0.94 |
| 6 | C | sae_reconstruction_direction | relu | 2042.0 | -1 | cos_mass_mean=0.5543 | 0.1094 | 0.0849 | 0.1327 | 0.0882 | 0.1764 |  |
| 6 | C | sae_top1_direction | relu | 2042.0 | 1 | cos_mass_mean=0.1737 | -0.0136 | -0.0271 | -0.0 | 0.5258 | 0.9517 | 0.9983 |
| 6 | C | sae_top1_direction | relu | 2042.0 | -1 | cos_mass_mean=0.1737 | 0.0295 | 0.0152 | 0.0432 | 0.3444 | 0.6889 |  |
| 6 | C | sae_top10_direction | relu | 2042.0 | 1 | cos_mass_mean=0.3872 | -0.0228 | -0.0437 | -0.0012 | 0.5591 | 0.8852 | 0.9848 |
| 6 | C | sae_top10_direction | relu | 2042.0 | -1 | cos_mass_mean=0.3872 | 0.0484 | 0.027 | 0.0699 | 0.2196 | 0.4393 |  |
| 6 | C | sae_reconstruction_direction | topk | 42.0 | 1 | cos_mass_mean=0.7365 | -0.062 | -0.0811 | -0.043 | 0.7903 | 0.4226 | 0.9749 |
| 6 | C | sae_reconstruction_direction | topk | 42.0 | -1 | cos_mass_mean=0.7365 | 0.0711 | 0.05 | 0.092 | 0.1764 | 0.3527 |  |
| 6 | C | sae_top1_direction | topk | 42.0 | 1 | cos_mass_mean=0.2073 | -0.0548 | -0.0714 | -0.0385 | 0.7288 | 0.5458 | 0.9749 |
| 6 | C | sae_top1_direction | topk | 42.0 | -1 | cos_mass_mean=0.2073 | 0.0305 | 0.0158 | 0.0458 | 0.3644 | 0.7288 |  |
| 6 | C | sae_top10_direction | topk | 42.0 | 1 | cos_mass_mean=0.4670 | 0.0132 | -0.0042 | 0.0302 | 0.4309 | 0.8619 | 0.9848 |
| 6 | C | sae_top10_direction | topk | 42.0 | -1 | cos_mass_mean=0.4670 | -0.0123 | -0.0294 | 0.0055 | 0.5424 | 0.9185 |  |
| 6 | C | sae_reconstruction_direction | topk | 1042.0 | 1 | cos_mass_mean=0.7343 | -0.0312 | -0.0488 | -0.014 | 0.6423 | 0.7188 | 0.9749 |
| 6 | C | sae_reconstruction_direction | topk | 1042.0 | -1 | cos_mass_mean=0.7343 | 0.0346 | 0.0164 | 0.0535 | 0.3278 | 0.6556 |  |
| 6 | C | sae_top1_direction | topk | 1042.0 | 1 | cos_mass_mean=0.2331 | -0.0566 | -0.0718 | -0.04 | 0.787 | 0.4293 | 0.9749 |
| 6 | C | sae_top1_direction | topk | 1042.0 | -1 | cos_mass_mean=0.2331 | 0.0577 | 0.0413 | 0.0733 | 0.213 | 0.426 |  |
| 6 | C | sae_top10_direction | topk | 1042.0 | 1 | cos_mass_mean=0.4919 | -0.049 | -0.0679 | -0.0305 | 0.7105 | 0.5824 | 0.9749 |
| 6 | C | sae_top10_direction | topk | 1042.0 | -1 | cos_mass_mean=0.4919 | 0.0621 | 0.0421 | 0.083 | 0.2446 | 0.4892 |  |
| 6 | C | sae_reconstruction_direction | topk | 2042.0 | 1 | cos_mass_mean=0.7490 | 0.0202 | 0.0042 | 0.0361 | 0.3894 | 0.7787 | 0.9779 |
| 6 | C | sae_reconstruction_direction | topk | 2042.0 | -1 | cos_mass_mean=0.7490 | -0.0173 | -0.033 | -0.0016 | 0.5757 | 0.8519 |  |
| 6 | C | sae_top1_direction | topk | 2042.0 | 1 | cos_mass_mean=0.2252 | -0.0705 | -0.0839 | -0.0566 | 0.8003 | 0.4027 | 0.9749 |
| 6 | C | sae_top1_direction | topk | 2042.0 | -1 | cos_mass_mean=0.2252 | 0.0756 | 0.0612 | 0.0895 | 0.183 | 0.3661 |  |
| 6 | C | sae_top10_direction | topk | 2042.0 | 1 | cos_mass_mean=0.4675 | -0.0339 | -0.0489 | -0.0189 | 0.6206 | 0.7621 | 0.9779 |
| 6 | C | sae_top10_direction | topk | 2042.0 | -1 | cos_mass_mean=0.4675 | 0.0257 | 0.0116 | 0.0399 | 0.3777 | 0.7554 |  |
| 6 | C | sae_reconstruction_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.5476 | -0.0204 | -0.0317 | -0.0088 | 0.584 | 0.8353 | 0.9848 |
| 6 | C | sae_reconstruction_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.5476 | 0.0252 | 0.0128 | 0.038 | 0.3577 | 0.7155 |  |
| 6 | C | sae_top1_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.0788 | 0.0278 | 0.0118 | 0.0444 | 0.3411 | 0.6822 | 0.9749 |
| 6 | C | sae_top1_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.0788 | -0.0102 | -0.0237 | 0.0024 | 0.5724 | 0.8586 |  |
| 6 | C | sae_top10_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.4208 | 0.0375 | 0.018 | 0.0557 | 0.3344 | 0.6689 | 0.9749 |
| 6 | C | sae_top10_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.4208 | -0.0443 | -0.0636 | -0.0232 | 0.7438 | 0.5158 |  |
| 6 | C | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.6121 | -0.035 | -0.052 | -0.0185 | 0.6622 | 0.6789 | 0.9749 |
| 6 | C | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.6121 | 0.0407 | 0.0223 | 0.0599 | 0.3028 | 0.6057 |  |
| 6 | C | sae_top1_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.1761 | -0.0182 | -0.0346 | -0.0012 | 0.5591 | 0.8852 | 0.9848 |
| 6 | C | sae_top1_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.1761 | -0.0001 | -0.0163 | 0.0155 | 0.4526 | 0.9052 |  |
| 6 | C | sae_top10_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.4509 | -0.0058 | -0.0244 | 0.0118 | 0.4842 | 0.9684 | 0.9983 |
| 6 | C | sae_top10_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.4509 | -0.0152 | -0.0329 | 0.0042 | 0.5308 | 0.9418 |  |
| 6 | C | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.5580 | -0.0358 | -0.0515 | -0.0205 | 0.6672 | 0.6689 | 0.9749 |
| 6 | C | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.5580 | 0.0362 | 0.0211 | 0.0515 | 0.3211 | 0.6423 |  |
| 6 | C | sae_top1_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.1620 | -0.0124 | -0.0317 | 0.0067 | 0.5042 | 0.995 | 0.9983 |
| 6 | C | sae_top1_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.1620 | -0.0196 | -0.0349 | -0.0044 | 0.5341 | 0.9351 |  |
| 6 | C | sae_top10_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.4093 | -0.0043 | -0.0191 | 0.0106 | 0.5125 | 0.9784 | 0.9983 |
| 6 | C | sae_top10_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.4093 | 0.0047 | -0.0089 | 0.0183 | 0.4725 | 0.9451 |  |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=0.01 | 0.044 | 0.0427 | 0.0453 | 0.0017 | 0.0033 | 0.0017 |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=0.02 | 0.088 | 0.0853 | 0.0905 | 0.0017 | 0.0033 | 0.0017 |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=0.05 | 0.2196 | 0.2129 | 0.2259 | 0.0017 | 0.0033 | 0.0017 |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=0.1 | 0.4383 | 0.4248 | 0.4509 | 0.0017 | 0.0033 | 0.0017 |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=0.25 | 1.0896 | 1.0557 | 1.1212 | 0.0017 | 0.0033 | 0.0017 |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=0.5 | 2.1675 | 2.0979 | 2.2345 | 0.0017 | 0.0033 | 0.0017 |
| 8 | A | unembedding_contrast |  |  | 0 | lambda=1.0 | 4.325 | 4.1743 | 4.4657 | 0.0017 | 0.0033 | 0.0017 |
| 8 | C | mass_mean_direction |  |  | 1 |  | 0.1287 | 0.0672 | 0.1917 | 0.188 | 0.376 | 0.9318 |
| 8 | C | mass_mean_direction |  |  | -1 |  | -0.1762 | -0.2384 | -0.1163 | 0.7005 | 0.6023 |  |
| 8 | C | logistic_probe_direction |  |  | 1 |  | -0.352 | -0.4329 | -0.2664 | 0.9118 | 0.1797 | 0.9468 |
| 8 | C | logistic_probe_direction |  |  | -1 |  | 0.2996 | 0.2145 | 0.3813 | 0.0433 | 0.0865 |  |
| 8 | C | sae_reconstruction_direction | relu | 42.0 | 1 | cos_mass_mean=0.6147 | -0.202 | -0.2611 | -0.1449 | 0.7354 | 0.5324 | 0.9749 |
| 8 | C | sae_reconstruction_direction | relu | 42.0 | -1 | cos_mass_mean=0.6147 | 0.1301 | 0.0799 | 0.1816 | 0.1864 | 0.3727 |  |
| 8 | C | sae_top1_direction | relu | 42.0 | 1 | cos_mass_mean=0.1915 | -0.0757 | -0.1291 | -0.0197 | 0.584 | 0.8353 | 0.9848 |
| 8 | C | sae_top1_direction | relu | 42.0 | -1 | cos_mass_mean=0.1915 | 0.1056 | 0.0577 | 0.1529 | 0.2097 | 0.4193 |  |
| 8 | C | sae_top10_direction | relu | 42.0 | 1 | cos_mass_mean=0.5031 | -0.0486 | -0.0938 | -0.0034 | 0.5108 | 0.9817 | 0.9983 |
| 8 | C | sae_top10_direction | relu | 42.0 | -1 | cos_mass_mean=0.5031 | -0.0684 | -0.1124 | -0.0247 | 0.5374 | 0.9285 |  |
| 8 | C | sae_reconstruction_direction | relu | 1042.0 | 1 | cos_mass_mean=0.6172 | 0.0092 | -0.0469 | 0.0673 | 0.3611 | 0.7221 | 0.9749 |
| 8 | C | sae_reconstruction_direction | relu | 1042.0 | -1 | cos_mass_mean=0.6172 | -0.051 | -0.1031 | 0.0036 | 0.4775 | 0.9551 |  |
| 8 | C | sae_top1_direction | relu | 1042.0 | 1 | cos_mass_mean=0.2371 | -0.0526 | -0.1111 | 0.006 | 0.4992 | 0.9983 | 0.9983 |
| 8 | C | sae_top1_direction | relu | 1042.0 | -1 | cos_mass_mean=0.2371 | -0.0274 | -0.0908 | 0.033 | 0.4493 | 0.8985 |  |
| 8 | C | sae_top10_direction | relu | 1042.0 | 1 | cos_mass_mean=0.4678 | -0.1486 | -0.2127 | -0.0859 | 0.6656 | 0.6722 | 0.9749 |
| 8 | C | sae_top10_direction | relu | 1042.0 | -1 | cos_mass_mean=0.4678 | 0.0975 | 0.0291 | 0.166 | 0.2945 | 0.589 |  |
| 8 | C | sae_reconstruction_direction | relu | 2042.0 | 1 | cos_mass_mean=0.6009 | 0.2239 | 0.1561 | 0.2965 | 0.0982 | 0.1963 | 0.8927 |
| 8 | C | sae_reconstruction_direction | relu | 2042.0 | -1 | cos_mass_mean=0.6009 | -0.2964 | -0.3731 | -0.2208 | 0.8519 | 0.2995 |  |
| 8 | C | sae_top1_direction | relu | 2042.0 | 1 | cos_mass_mean=0.0765 | -0.0919 | -0.1478 | -0.0336 | 0.5458 | 0.9118 | 0.9848 |
| 8 | C | sae_top1_direction | relu | 2042.0 | -1 | cos_mass_mean=0.0765 | 0.1687 | 0.1136 | 0.2235 | 0.1398 | 0.2795 |  |
| 8 | C | sae_top10_direction | relu | 2042.0 | 1 | cos_mass_mean=0.4617 | 0.3564 | 0.275 | 0.4439 | 0.0483 | 0.0965 | 0.7445 |
| 8 | C | sae_top10_direction | relu | 2042.0 | -1 | cos_mass_mean=0.4617 | -0.6249 | -0.7424 | -0.5095 | 0.9933 | 0.0166 |  |
| 8 | C | sae_reconstruction_direction | topk | 42.0 | 1 | cos_mass_mean=0.7117 | -0.0799 | -0.1437 | -0.016 | 0.5275 | 0.9484 | 0.9983 |
| 8 | C | sae_reconstruction_direction | topk | 42.0 | -1 | cos_mass_mean=0.7117 | 0.0291 | -0.0284 | 0.0867 | 0.3311 | 0.6622 |  |
| 8 | C | sae_top1_direction | topk | 42.0 | 1 | cos_mass_mean=0.1539 | -0.3306 | -0.4091 | -0.2512 | 0.8802 | 0.2429 | 0.8927 |
| 8 | C | sae_top1_direction | topk | 42.0 | -1 | cos_mass_mean=0.1539 | 0.2927 | 0.2157 | 0.3729 | 0.0566 | 0.1131 |  |
| 8 | C | sae_top10_direction | topk | 42.0 | 1 | cos_mass_mean=0.4167 | -0.2009 | -0.262 | -0.1392 | 0.7471 | 0.5092 | 0.9749 |
| 8 | C | sae_top10_direction | topk | 42.0 | -1 | cos_mass_mean=0.4167 | 0.1773 | 0.113 | 0.2398 | 0.183 | 0.3661 |  |
| 8 | C | sae_reconstruction_direction | topk | 1042.0 | 1 | cos_mass_mean=0.7563 | -0.0348 | -0.1015 | 0.0308 | 0.4476 | 0.8952 | 0.9848 |
| 8 | C | sae_reconstruction_direction | topk | 1042.0 | -1 | cos_mass_mean=0.7563 | -0.0672 | -0.1308 | -0.0032 | 0.5075 | 0.9884 |  |
| 8 | C | sae_top1_direction | topk | 1042.0 | 1 | cos_mass_mean=0.2278 | -0.3348 | -0.4198 | -0.2516 | 0.8852 | 0.2329 | 0.8927 |
| 8 | C | sae_top1_direction | topk | 1042.0 | -1 | cos_mass_mean=0.2278 | 0.0087 | -0.0387 | 0.057 | 0.4093 | 0.8186 |  |
| 8 | C | sae_top10_direction | topk | 1042.0 | 1 | cos_mass_mean=0.4314 | 0.0221 | -0.0476 | 0.0914 | 0.381 | 0.7621 | 0.9779 |
| 8 | C | sae_top10_direction | topk | 1042.0 | -1 | cos_mass_mean=0.4314 | -0.2646 | -0.3394 | -0.1862 | 0.8419 | 0.3195 |  |
| 8 | C | sae_reconstruction_direction | topk | 2042.0 | 1 | cos_mass_mean=0.7195 | 0.0748 | 0.0083 | 0.1374 | 0.2612 | 0.5225 | 0.9749 |
| 8 | C | sae_reconstruction_direction | topk | 2042.0 | -1 | cos_mass_mean=0.7195 | -0.1332 | -0.1992 | -0.0654 | 0.6389 | 0.7255 |  |
| 8 | C | sae_top1_direction | topk | 2042.0 | 1 | cos_mass_mean=0.1573 | -0.2629 | -0.369 | -0.1649 | 0.8502 | 0.3028 | 0.94 |
| 8 | C | sae_top1_direction | topk | 2042.0 | -1 | cos_mass_mean=0.1573 | 0.0335 | -0.0177 | 0.0841 | 0.3511 | 0.7022 |  |
| 8 | C | sae_top10_direction | topk | 2042.0 | 1 | cos_mass_mean=0.4401 | 0.2469 | 0.1741 | 0.3188 | 0.0982 | 0.1963 | 0.8927 |
| 8 | C | sae_top10_direction | topk | 2042.0 | -1 | cos_mass_mean=0.4401 | -0.445 | -0.5512 | -0.3392 | 0.9434 | 0.1165 |  |
| 8 | C | sae_reconstruction_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.6229 | -0.2404 | -0.3323 | -0.1553 | 0.787 | 0.4293 | 0.9749 |
| 8 | C | sae_reconstruction_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.6229 | 0.2828 | 0.1975 | 0.3709 | 0.0449 | 0.0899 |  |
| 8 | C | sae_top1_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.2208 | -0.1392 | -0.1817 | -0.0949 | 0.6423 | 0.7188 | 0.9749 |
| 8 | C | sae_top1_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.2208 | 0.0461 | -0.0104 | 0.1019 | 0.3544 | 0.7088 |  |
| 8 | C | sae_top10_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.4476 | -0.3293 | -0.413 | -0.249 | 0.8586 | 0.2862 | 0.94 |
| 8 | C | sae_top10_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.4476 | 0.279 | 0.2018 | 0.3579 | 0.0948 | 0.1897 |  |
| 8 | C | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.6209 | 0.0454 | -0.0306 | 0.1177 | 0.3028 | 0.6057 | 0.9749 |
| 8 | C | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.6209 | -0.0287 | -0.0948 | 0.0404 | 0.4359 | 0.8719 |  |
| 8 | C | sae_top1_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.1593 | -0.0499 | -0.1018 | -0.0009 | 0.5607 | 0.8819 | 0.9848 |
| 8 | C | sae_top1_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.1593 | -0.0431 | -0.1006 | 0.0159 | 0.5458 | 0.9118 |  |
| 8 | C | sae_top10_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.4460 | -0.0651 | -0.146 | 0.0184 | 0.5574 | 0.8885 | 0.9848 |
| 8 | C | sae_top10_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.4460 | 0.157 | 0.0826 | 0.2345 | 0.2263 | 0.4526 |  |
| 8 | C | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.6404 | 0.0514 | -0.0064 | 0.1062 | 0.2995 | 0.599 | 0.9749 |
| 8 | C | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.6404 | -0.1592 | -0.2176 | -0.0978 | 0.6722 | 0.6589 |  |
| 8 | C | sae_top1_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.0968 | -0.4168 | -0.5135 | -0.319 | 0.9285 | 0.1464 | 0.8785 |
| 8 | C | sae_top1_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.0968 | 0.3125 | 0.2172 | 0.4079 | 0.0399 | 0.0799 |  |
| 8 | C | sae_top10_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.4015 | -0.1417 | -0.1959 | -0.0898 | 0.6705 | 0.6622 | 0.9749 |
| 8 | C | sae_top10_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.4015 | 0.0236 | -0.0294 | 0.0774 | 0.3561 | 0.7121 |  |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=0.01 | 0.0859 | 0.0834 | 0.0884 | 0.0017 | 0.0033 | 0.0017 |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=0.02 | 0.1718 | 0.1668 | 0.1767 | 0.0017 | 0.0033 | 0.0017 |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=0.05 | 0.4289 | 0.4165 | 0.4413 | 0.0017 | 0.0033 | 0.0017 |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=0.1 | 0.8562 | 0.8315 | 0.881 | 0.0017 | 0.0033 | 0.0017 |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=0.25 | 2.128 | 2.0661 | 2.1905 | 0.0017 | 0.0033 | 0.0017 |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=0.5 | 4.2122 | 4.0864 | 4.3384 | 0.0017 | 0.0033 | 0.0017 |
| 10 | A | unembedding_contrast |  |  | 0 | lambda=1.0 | 8.2413 | 7.9911 | 8.4975 | 0.0017 | 0.0033 | 0.0017 |
| 10 | C | mass_mean_direction |  |  | 1 |  | 0.0803 | -0.0361 | 0.1888 | 0.2329 | 0.4659 | 0.9318 |
| 10 | C | mass_mean_direction |  |  | -1 |  | -0.1454 | -0.259 | -0.0228 | 0.7438 | 0.5158 |  |
| 10 | C | logistic_probe_direction |  |  | 1 |  | -0.1695 | -0.2303 | -0.1053 | 0.7887 | 0.426 | 0.9468 |
| 10 | C | logistic_probe_direction |  |  | -1 |  | 0.032 | -0.0334 | 0.0958 | 0.3128 | 0.6256 |  |
| 10 | C | sae_reconstruction_direction | relu | 42.0 | 1 | cos_mass_mean=0.7122 | -0.1058 | -0.2404 | 0.0328 | 0.6439 | 0.7155 | 0.9749 |
| 10 | C | sae_reconstruction_direction | relu | 42.0 | -1 | cos_mass_mean=0.7122 | 0.0173 | -0.1247 | 0.1596 | 0.3461 | 0.6922 |  |
| 10 | C | sae_top1_direction | relu | 42.0 | 1 | cos_mass_mean=0.1939 | 0.0105 | -0.0759 | 0.0991 | 0.3245 | 0.6489 | 0.9749 |
| 10 | C | sae_top1_direction | relu | 42.0 | -1 | cos_mass_mean=0.1939 | -0.0502 | -0.1022 | 0.0044 | 0.4859 | 0.9717 |  |
| 10 | C | sae_top10_direction | relu | 42.0 | 1 | cos_mass_mean=0.4420 | -0.127 | -0.2989 | 0.0499 | 0.807 | 0.3894 | 0.9749 |
| 10 | C | sae_top10_direction | relu | 42.0 | -1 | cos_mass_mean=0.4420 | 0.0357 | -0.1446 | 0.2093 | 0.411 | 0.822 |  |
| 10 | C | sae_reconstruction_direction | relu | 1042.0 | 1 | cos_mass_mean=0.6907 | -0.0895 | -0.2059 | 0.0377 | 0.6156 | 0.772 | 0.9779 |
| 10 | C | sae_reconstruction_direction | relu | 1042.0 | -1 | cos_mass_mean=0.6907 | 0.0184 | -0.1118 | 0.1444 | 0.3428 | 0.6855 |  |
| 10 | C | sae_top1_direction | relu | 1042.0 | 1 | cos_mass_mean=0.1700 | 0.0814 | -0.0179 | 0.1783 | 0.1913 | 0.3827 | 0.9749 |
| 10 | C | sae_top1_direction | relu | 1042.0 | -1 | cos_mass_mean=0.1700 | -0.1092 | -0.1614 | -0.0558 | 0.6672 | 0.6689 |  |
| 10 | C | sae_top10_direction | relu | 1042.0 | 1 | cos_mass_mean=0.4811 | -0.0102 | -0.192 | 0.1795 | 0.4426 | 0.8852 | 0.9848 |
| 10 | C | sae_top10_direction | relu | 1042.0 | -1 | cos_mass_mean=0.4811 | -0.1052 | -0.307 | 0.0851 | 0.6972 | 0.609 |  |
| 10 | C | sae_reconstruction_direction | relu | 2042.0 | 1 | cos_mass_mean=0.6137 | -0.0659 | -0.1816 | 0.0496 | 0.5507 | 0.9018 | 0.9848 |
| 10 | C | sae_reconstruction_direction | relu | 2042.0 | -1 | cos_mass_mean=0.6137 | 0.0288 | -0.0917 | 0.1489 | 0.3178 | 0.6356 |  |
| 10 | C | sae_top1_direction | relu | 2042.0 | 1 | cos_mass_mean=0.1407 | -0.18 | -0.2549 | -0.1083 | 0.8253 | 0.3527 | 0.9524 |
| 10 | C | sae_top1_direction | relu | 2042.0 | -1 | cos_mass_mean=0.1407 | 0.16 | 0.0819 | 0.2387 | 0.1115 | 0.223 |  |
| 10 | C | sae_top10_direction | relu | 2042.0 | 1 | cos_mass_mean=0.3923 | -0.0543 | -0.2235 | 0.1172 | 0.6622 | 0.6789 | 0.9749 |
| 10 | C | sae_top10_direction | relu | 2042.0 | -1 | cos_mass_mean=0.3923 | 0.0159 | -0.1546 | 0.1799 | 0.5008 | 1.0 |  |
| 10 | C | sae_reconstruction_direction | topk | 42.0 | 1 | cos_mass_mean=0.7894 | 0.016 | -0.0698 | 0.0978 | 0.3461 | 0.6922 | 0.9749 |
| 10 | C | sae_reconstruction_direction | topk | 42.0 | -1 | cos_mass_mean=0.7894 | -0.0926 | -0.1807 | -0.0026 | 0.619 | 0.7654 |  |
| 10 | C | sae_top1_direction | topk | 42.0 | 1 | cos_mass_mean=0.2348 | -0.2927 | -0.3587 | -0.2298 | 0.9651 | 0.0732 | 0.6589 |
| 10 | C | sae_top1_direction | topk | 42.0 | -1 | cos_mass_mean=0.2348 | 0.2517 | 0.19 | 0.317 | 0.0449 | 0.0899 |  |
| 10 | C | sae_top10_direction | topk | 42.0 | 1 | cos_mass_mean=0.4587 | -0.2892 | -0.3894 | -0.1886 | 0.9384 | 0.1265 | 0.8034 |
| 10 | C | sae_top10_direction | topk | 42.0 | -1 | cos_mass_mean=0.4587 | 0.2445 | 0.1549 | 0.3367 | 0.0449 | 0.0899 |  |
| 10 | C | sae_reconstruction_direction | topk | 1042.0 | 1 | cos_mass_mean=0.7526 | 0.1628 | 0.0741 | 0.2447 | 0.1281 | 0.2562 | 0.8927 |
| 10 | C | sae_reconstruction_direction | topk | 1042.0 | -1 | cos_mass_mean=0.7526 | -0.1908 | -0.2748 | -0.1002 | 0.827 | 0.3494 |  |
| 10 | C | sae_top1_direction | topk | 1042.0 | 1 | cos_mass_mean=0.2361 | -0.1064 | -0.1688 | -0.0451 | 0.7005 | 0.6023 | 0.9749 |
| 10 | C | sae_top1_direction | topk | 1042.0 | -1 | cos_mass_mean=0.2361 | 0.0491 | -0.0171 | 0.1165 | 0.2562 | 0.5125 |  |
| 10 | C | sae_top10_direction | topk | 1042.0 | 1 | cos_mass_mean=0.4377 | 0.3 | 0.2256 | 0.3786 | 0.0283 | 0.0566 | 0.6589 |
| 10 | C | sae_top10_direction | topk | 1042.0 | -1 | cos_mass_mean=0.4377 | -0.2965 | -0.3796 | -0.2174 | 0.9584 | 0.0865 |  |
| 10 | C | sae_reconstruction_direction | topk | 2042.0 | 1 | cos_mass_mean=0.7742 | 0.2029 | 0.113 | 0.2885 | 0.0899 | 0.1797 | 0.8927 |
| 10 | C | sae_reconstruction_direction | topk | 2042.0 | -1 | cos_mass_mean=0.7742 | -0.2164 | -0.3124 | -0.1168 | 0.8719 | 0.2596 |  |
| 10 | C | sae_top1_direction | topk | 2042.0 | 1 | cos_mass_mean=0.2168 | 0.3881 | 0.2801 | 0.4904 | 0.0333 | 0.0666 | 0.6589 |
| 10 | C | sae_top1_direction | topk | 2042.0 | -1 | cos_mass_mean=0.2168 | -0.379 | -0.466 | -0.2824 | 0.9734 | 0.0566 |  |
| 10 | C | sae_top10_direction | topk | 2042.0 | 1 | cos_mass_mean=0.4509 | 0.3907 | 0.3074 | 0.4754 | 0.01 | 0.02 | 0.599 |
| 10 | C | sae_top10_direction | topk | 2042.0 | -1 | cos_mass_mean=0.4509 | -0.367 | -0.4542 | -0.2781 | 0.9767 | 0.0499 |  |
| 10 | C | sae_reconstruction_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.6964 | 0.3 | 0.1799 | 0.4144 | 0.0266 | 0.0532 | 0.6589 |
| 10 | C | sae_reconstruction_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.6964 | -0.3023 | -0.4241 | -0.1787 | 0.9617 | 0.0799 |  |
| 10 | C | sae_top1_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.1363 | 0.0609 | 0.0101 | 0.1132 | 0.2512 | 0.5025 | 0.9749 |
| 10 | C | sae_top1_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.1363 | -0.128 | -0.1848 | -0.0728 | 0.7188 | 0.5657 |  |
| 10 | C | sae_top10_direction | jumprelu | 42.0 | 1 | cos_mass_mean=0.4502 | 0.0114 | -0.1772 | 0.1995 | 0.4143 | 0.8286 | 0.9848 |
| 10 | C | sae_top10_direction | jumprelu | 42.0 | -1 | cos_mass_mean=0.4502 | -0.0879 | -0.2709 | 0.0934 | 0.6323 | 0.7388 |  |
| 10 | C | sae_reconstruction_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.6963 | 0.1776 | 0.0935 | 0.2584 | 0.1115 | 0.223 | 0.8927 |
| 10 | C | sae_reconstruction_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.6963 | -0.1921 | -0.2746 | -0.1055 | 0.8303 | 0.3428 |  |
| 10 | C | sae_top1_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.2156 | 0.1749 | 0.104 | 0.2449 | 0.1065 | 0.213 | 0.8927 |
| 10 | C | sae_top1_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.2156 | -0.2177 | -0.2845 | -0.148 | 0.8436 | 0.3161 |  |
| 10 | C | sae_top10_direction | jumprelu | 1042.0 | 1 | cos_mass_mean=0.5082 | 0.0638 | -0.0144 | 0.1437 | 0.2496 | 0.4992 | 0.9749 |
| 10 | C | sae_top10_direction | jumprelu | 1042.0 | -1 | cos_mass_mean=0.5082 | -0.1177 | -0.2034 | -0.0303 | 0.6622 | 0.6789 |  |
| 10 | C | sae_reconstruction_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.7021 | 0.1648 | 0.0583 | 0.2703 | 0.1248 | 0.2496 | 0.8927 |
| 10 | C | sae_reconstruction_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.7021 | -0.2212 | -0.3251 | -0.1131 | 0.8785 | 0.2463 |  |
| 10 | C | sae_top1_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.2666 | -0.2588 | -0.3574 | -0.1643 | 0.9035 | 0.1963 | 0.8927 |
| 10 | C | sae_top1_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.2666 | 0.2464 | 0.1451 | 0.3461 | 0.0566 | 0.1131 |  |
| 10 | C | sae_top10_direction | jumprelu | 2042.0 | 1 | cos_mass_mean=0.4949 | -0.0917 | -0.1865 | 0.0038 | 0.6057 | 0.792 | 0.9832 |
| 10 | C | sae_top10_direction | jumprelu | 2042.0 | -1 | cos_mass_mean=0.4949 | 0.006 | -0.0818 | 0.0997 | 0.3993 | 0.7987 |  |

## OWT loss recovered vs mean ablation, public and in-house SAEs (validation)

| group | layer | loss_recovered_pct | owt__kl_mean |
|---|---|---|---|
| jumprelu | 0 | 95.4677 | 0.3109 |
| jumprelu | 1 | 99.8962 | 0.0153 |
| jumprelu | 2 | 99.684 | 0.032 |
| jumprelu | 3 | 99.3513 | 0.0541 |
| jumprelu | 4 | 99.2014 | 0.0751 |
| jumprelu | 5 | 98.2306 | 0.104 |
| jumprelu | 6 | 98.2378 | 0.1078 |
| jumprelu | 7 | 98.0139 | 0.1216 |
| jumprelu | 8 | 97.7683 | 0.136 |
| jumprelu | 9 | 97.2796 | 0.1538 |
| jumprelu | 10 | 96.8835 | 0.1776 |
| jumprelu | 11 | 95.5908 | 0.2317 |
| public_relu | 0 | 99.6591 | 0.0233 |
| public_relu | 1 | 99.8126 | 0.0259 |
| public_relu | 2 | 99.1054 | 0.081 |
| public_relu | 3 | 97.3987 | 0.1823 |
| public_relu | 4 | 98.1532 | 0.1543 |
| public_relu | 5 | 97.0705 | 0.1681 |
| public_relu | 6 | 97.1034 | 0.1688 |
| public_relu | 7 | 96.7939 | 0.18 |
| public_relu | 8 | 96.8044 | 0.1826 |
| public_relu | 9 | 96.5496 | 0.1918 |
| public_relu | 10 | 95.589 | 0.2488 |
| public_relu | 11 | 94.5708 | 0.2826 |
| public_topk | 1 | 99.87 | 0.0139 |
| public_topk | 2 | 99.5541 | 0.0339 |
| public_topk | 3 | 99.2621 | 0.0575 |
| public_topk | 4 | 99.0905 | 0.075 |
| public_topk | 5 | 98.0658 | 0.1056 |
| public_topk | 6 | 98.0463 | 0.1135 |
| public_topk | 7 | 97.5486 | 0.1363 |
| public_topk | 8 | 97.1313 | 0.1539 |
| public_topk | 9 | 96.5138 | 0.1757 |
| public_topk | 10 | 96.2769 | 0.1956 |
| public_topk | 11 | 95.2033 | 0.2333 |
| relu | 0 | 99.5564 | 0.0303 |
| relu | 1 | 99.7767 | 0.0231 |
| relu | 2 | 99.4501 | 0.0491 |
| relu | 3 | 98.7267 | 0.094 |
| relu | 4 | 98.4294 | 0.1298 |
| relu | 5 | 96.7444 | 0.1808 |
| relu | 6 | 96.637 | 0.1922 |
| relu | 7 | 96.2784 | 0.2174 |
| relu | 8 | 95.7582 | 0.2431 |
| relu | 9 | 95.1228 | 0.2713 |
| relu | 10 | 94.6908 | 0.2947 |
| relu | 11 | 92.9714 | 0.3685 |
| topk | 0 | 99.1187 | 0.0617 |
| topk | 1 | 99.7014 | 0.0288 |
| topk | 2 | 99.4323 | 0.0462 |
| topk | 3 | 99.0719 | 0.0673 |
| topk | 4 | 98.973 | 0.0851 |
| topk | 5 | 97.9577 | 0.1132 |
| topk | 6 | 98.017 | 0.117 |
| topk | 7 | 97.8512 | 0.1289 |
| topk | 8 | 97.6049 | 0.1439 |
| topk | 9 | 97.0721 | 0.161 |
| topk | 10 | 96.8154 | 0.176 |
| topk | 11 | 95.8195 | 0.2177 |
