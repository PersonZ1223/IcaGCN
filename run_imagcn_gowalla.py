from recbole.quick_start import run_recbole



parameter_dict = {
#   'neg_sampling': None,
}

config_file_list=['gowalla.yaml']
run_recbole(model='IMAGCN', dataset='gowalla', config_file_list=config_file_list, config_dict=parameter_dict)
