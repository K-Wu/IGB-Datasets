mkdir -p igb260m
cd igb260m

# paper
mkdir -p paper
cd paper
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_feat.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_19.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/paper_id_index_mapping.npy
cd ..

# paper__cites__paper
mkdir -p paper__cites__paper
cd paper__cites__paper
wget https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__cites__paper/edge_index.npy
cd ..

# Leaving igb260m
cd ..