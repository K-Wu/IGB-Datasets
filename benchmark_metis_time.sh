python -m benchmark.do_partition_graph --num_parts=2 --num_trainers_per_machine=2 --dataset=igb240m_medium >igb240m_medium_2_2_metis_output.log 2>&1
rm -rf out_igb240m_medium_2_2_with_metis
python -m benchmark.do_partition_graph --num_parts=2 --num_trainers_per_machine=4 --dataset=igb240m_medium >igb240m_medium_2_4_metis_output.log 2>&1
rm -rf out_igb240m_medium_2_4_with_metis
python -m benchmark.do_partition_graph --num_parts=4 --num_trainers_per_machine=2 --dataset=igb240m_medium >igb240m_medium_4_2_metis_output.log 2>&1
rm -rf out_igb240m_medium_4_2_with_metis
python -m benchmark.do_partition_graph --num_parts=4 --num_trainers_per_machine=4 --dataset=igb240m_medium >igb240m_medium_4_4_metis_output.log 2>&1
rm -rf out_igb240m_medium_4_4_with_metis
python -m benchmark.do_partition_graph --num_parts=8 --num_trainers_per_machine=4 --dataset=igb240m_medium >igb240m_medium_8_4_metis_output.log 2>&1
rm -rf out_igb240m_medium_8_4_with_metis