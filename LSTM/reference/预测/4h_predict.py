from config import *
from model import predict

if __name__ == '__main__':
    #电负荷4h预测
    predict(predict_data_path = cold_load_4h_pre_data_path,
            result_path = cold_load_4h_result_data_path,
            model_path = cold_load_4h_model_path,
            feature_name = load_feature_name,
            predict_length = 4,
            hidden_size = 64)

    #热负荷4h预测
    predict(predict_data_path = heat_load_4h_pre_data_path,
            result_path = heat_load_4h_result_data_path,
            model_path = heat_load_4h_model_path,
            feature_name = load_feature_name,
            predict_length = 4,
            hidden_size = 64)

    #冷负荷4h预测
    predict(predict_data_path = ele_load_4h_pre_data_path,
            result_path = ele_load_4h_result_data_path,
            model_path = ele_load_4h_model_path,
            feature_name = load_feature_name,
            predict_length = 4,
            hidden_size = 64)

    #光伏4h预测
    predict(predict_data_path = PV_4h_pre_data_path,
            result_path = PV_4h_result_data_path,
            model_path = PV_4h_model_path,
            feature_name = PV_feature_name,
            predict_length = 4,
            hidden_size = 64)