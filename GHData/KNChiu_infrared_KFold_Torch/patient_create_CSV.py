import os
import pandas as pd




if __name__ == '__main__':
    # imgPathlist = []
    imgPath = r'G:\我的雲端硬碟\Lab\Project\外科溫度\醫師分享圖片\傷口分類'

    patient_DataFrame = {
        'image_name': [],      # 圖片檔名
        'class': [],           # 傷口類型
        'patient_id': [],      # 病人ID
        'affected_part': []    # 傷口部位
                        }


    for path, dir_list, file_list in os.walk(imgPath):  
        for file_name in file_list:
            if file_name.split(".")[-1] == "jpg":
                patient_class = ""
                if path.split("\\")[-1] == "Ischemia FLIR":
                    patient_class = "Ischemia"
                elif path.split("\\")[-1] == "感染恢復期" or  path.split("\\")[-1] == "感染急性期":
                    patient_class = "Infect"
                elif path.split("\\")[-1] == "混合型":
                    patient_class = "Mixed"

                patient_DataFrame['image_name'].append(file_name)
                patient_DataFrame['patient_id'].append("")
                patient_DataFrame['affected_part'].append("")
                patient_DataFrame['class'].append(patient_class)

    print(pd.DataFrame(patient_DataFrame))
    pd.DataFrame(patient_DataFrame).to_csv(r'patient_DataFrame.csv', index=False)