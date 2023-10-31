mkdir -p igbh_large
cd igbh_large

# paper
mkdir -p paper
wget -O paper/node_feat.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/node_feat.npy
wget -O paper/node_label_19.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/node_label_19.npy
wget -O paper/node_label_2K.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/node_label_2K.npy
wget -O paper/paper_id_index_mapping.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper/paper_id_index_mapping.npy

# paper__cites__paper
mkdir -p paper__cites__paper
cd paper__cites__paper
wget https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__cites__paper/edge_index.npy
cd ..

# paper__cites__paper
mkdir -p paper__cites__paper
wget -O paper__cites__paper/edge_index.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__cites__paper/edge_index.npy

# author
mkdir -p author
wget -O author/author_id_index_mapping.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/author/author_id_index_mapping.npy
wget -O author/node_feat.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/author/node_feat.npy

# conference
mkdir -p conference
wget -O conference/conference_id_index_mapping.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/conference/conference_id_index_mapping.npy
wget -O conference/node_feat.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/conference/node_feat.npy

# institute
mkdir -p institute
wget -O institute/institute_id_index_mapping.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/institute/institute_id_index_mapping.npy
wget -O institute/node_feat.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/institute/node_feat.npy

# journal
mkdir -p journal
wget -O journal/journal_id_index_mapping.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/journal/journal_id_index_mapping.npy
wget -O journal/node_feat.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/journal/node_feat.npy

# fos
mkdir -p fos
wget -O fos/fos_id_index_mapping.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/fos/fos_id_index_mapping.npy
wget -O fos/node_feat.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/fos/node_feat.npy

# author__affiliated_to__institute
mkdir -p author__affiliated_to__institute
wget -O author__affiliated_to__institute/edge_index.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/author__affiliated_to__institute/edge_index.npy

# paper__published__journal
mkdir -p paper__published__journal
wget -O paper__published__journal/edge_index.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__published__journal/edge_index.npy

# paper__topic__fos
mkdir -p paper__topic__fos
wget -O paper__topic__fos/edge_index.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__topic__fos/edge_index.npy

# paper__venue__conference
mkdir -p paper__venue__conference
wget -O paper__venue__conference/edge_index.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__venue__conference/edge_index.npy

# paper__written_by__author
mkdir -p paper__written_by__author
wget -O paper__written_by__author/edge_index.npy https://igb-public.s3.us-east-2.amazonaws.com/igb_large/processed/paper__written_by__author/edge_index.npy

# Leaving igbh_large
cd ..
