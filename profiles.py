MoEUT_44M = dict(
    d_model=412, n_layers=16, n_heads=4, ff_n_experts=155, att_n_experts=8, d_head=82, 
    ff_k=12, group_size=2)
    
MoEUT_126M = dict(
    d_model=768, n_layers=18, n_heads=4, ff_n_experts=254, att_n_experts=10, d_head=96, 
    ff_k=12, group_size=2)

MoEUT_244M = dict(
    d_model=1024, n_layers=18, n_heads=4, ff_n_experts=387, att_n_experts=10, d_head=128, 
    ff_k=16, group_size=2)

MoEUT_318M = dict(
    d_model=1024, n_layers=24, n_heads=4, ff_n_experts=338, att_n_experts=10, d_head=128, 
    ff_k=16, group_size=3)

MoEUT_727M = dict(
    d_model=1024, n_layers=36, n_heads=5, ff_n_experts=467, att_n_experts=13, d_head=128, 
    ff_k=20, group_size=4)

MoEUT_1B = dict(
    d_model=1536, n_layers=36, n_heads=6, ff_n_experts=565, att_n_experts=12, d_head=128, 
    ff_k=24, group_size=4)
