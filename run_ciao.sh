python run_GraphRec_example.py --dataset ciao --gpu_id 0
python run_GraphRec_example.py --dataset ciao --users_from_test_must_be_in_train True --gpu_id 0
python run_GraphRec_example.py --dataset ciao --is_remove_zero_ratings True --is_remove_duplicates True --gpu_id 0
python run_GraphRec_example.py --dataset ciao --is_remove_zero_ratings True --is_remove_duplicates True --users_from_test_must_be_in_train True --gpu_id 0
python run_GraphRec_example.py --dataset ciao --is_remove_zero_ratings True --is_remove_duplicates True --is_remove_users_with_no_ratings True --gpu_id 1
python run_GraphRec_example.py --dataset ciao --is_remove_zero_ratings True --is_remove_duplicates True --is_remove_users_with_no_ratings True --users_from_test_must_be_in_train True --gpu_id 1
python run_GraphRec_example.py --dataset ciao --is_remove_zero_ratings True --is_remove_duplicates True --is_remove_users_with_no_ratings True --is_remove_users_with_no_connections True --gpu_id 1
python run_GraphRec_example.py --dataset ciao --is_remove_zero_ratings True --is_remove_duplicates True --is_remove_users_with_no_ratings True --is_remove_users_with_no_connections True --users_from_test_must_be_in_train True --gpu_id 1
