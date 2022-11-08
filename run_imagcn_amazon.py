from recbole.quick_start import run_recbole



parameter_dict = {
#   'neg_sampling': None,
}

config_file_list=['amazon.yaml']
run_recbole(model='LightGCN', dataset='Amazon_Books', config_file_list=config_file_list, config_dict=parameter_dict)
