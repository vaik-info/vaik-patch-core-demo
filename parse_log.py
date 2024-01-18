import argparse
import glob
import os
from io import StringIO
import pandas as pd
def parse(log_path, columns, delimiter='---'):
    with open(log_path, 'r') as f:
        data = f.read()
    category = os.path.basename(log_path).split('_')[0]
    auroc_by_max_score_df = pd.read_csv(StringIO(data.split(delimiter)[0]), skiprows=1, skipinitialspace=True)
    auroc_by_mean_score_df = pd.read_csv(StringIO(data.split(delimiter)[2]), skiprows=2, skipinitialspace=True)
    pixelwise_auroc_df = pd.read_csv(StringIO(data.split(delimiter)[4]), skiprows=2, skipinitialspace=True)
    parse_df = pd.DataFrame(data=[[category,
                            auroc_by_mean_score_df.iloc[0]['instance_auroc'], pixelwise_auroc_df.iloc[0]['full_pixel_auroc'], pixelwise_auroc_df.iloc[0]['anomaly_pixel_auroc'],
                            auroc_by_mean_score_df.iloc[0]['reliable_good_ratio'], auroc_by_mean_score_df.iloc[0]['anomaly_min_score'],
                            auroc_by_max_score_df.iloc[0]['instance_auroc'], auroc_by_max_score_df.iloc[0]['reliable_good_ratio'], auroc_by_max_score_df.iloc[0]['anomaly_min_score'],
                            ]], columns=columns)

    return parse_df
def main(input_dir_path, output_csv_path):
    df = pd.DataFrame(data=None, columns=['category', 'instance_auroc_mean', 'full_pixel_auroc', 'anomaly_pixel_auroc', 'reliable_good_ratio_mean', 'anomaly_min_score_mean', 'instance_auroc_max', 'reliable_good_ratio_max', 'anomaly_min_score_max'])
    log_path_list = sorted(glob.glob(os.path.join(input_dir_path, '*.log')))
    for log_path in log_path_list:
        parse_df = parse(log_path, df.columns)
        df = pd.concat([df, parse_df], ignore_index=True)

    # calc mean
    mean_df = pd.DataFrame(data=[['mean', ] + df.drop(columns='category').mean().tolist()], columns=df.columns)
    df = pd.concat([mean_df, df], ignore_index=True)
    df = df.round(4)
    df.to_csv(output_csv_path, index=False)

    print(df.iloc[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--input_dir_path', type=str, default='~/Desktop/anomaly.v.0.0.1_experiment_log')
    parser.add_argument('--output_csv_path', type=str, default='~/Desktop/anomaly.v.0.0.1_experiment_log/log.csv')

    args = parser.parse_args()

    args.input_dir_path = os.path.expanduser(args.input_dir_path)
    args.output_csv_path = os.path.expanduser(args.output_csv_path)

    main(**args.__dict__)