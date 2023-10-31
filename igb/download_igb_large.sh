mkdir -p igb_large
cd igb_large

# paper
wget https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/node_feat.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/node_label_19.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/node_label_2K.npy
wget https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/paper_id_index_mapping.npy

# paper__cites__paper
wget --recursive --no-parent https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__cites__paper/edge_index.npy

# Leaving igb_large
cd ..