# PROJECT 1

### memo

기타 음형 데이터에서 클린 톤과 다양한 이펙터(드라이브, 리버브, 딜레이, 모듈레이션)를 구분하려면 각 이펙터가 소리에 미치는 특성을 잘 반영하는 여러 종류의 특징(feature)을 추출해야 합니다. 다음은 각 이펙터를 효과적으로 구분할 수 있는 주요 오디오 특징들입니다:

1. **멜-주파수 켑스트럼 계수 (MFCCs)**: 소리의 질감과 타이밍을 제외한 주파수 특성을 포착하여 각 이펙터의 고유한 주파수 응답을 식별할 수 있습니다.
2. **스펙트럴 센트로이드 (Spectral Centroid)**: 소리의 "밝기"나 "명료성"을 측정하여, 특히 모듈레이션 이펙터의 변조 효과를 감지하는 데 유용합니다.
3. **스펙트럴 롤오프 (Spectral Rolloff)**: 주파수 스펙트럼의 상위 부분에 대한 정보를 제공하여, 특히 딜레이와 리버브의 잔향 특성을 포착하는 데 도움이 됩니다.
4. **제로 크로싱 레이트 (Zero Crossing Rate, ZCR)**: 오디오 신호가 시간 축을 기준으로 얼마나 자주 부호를 변경하는지 측정하여, 특히 드라이브 같은 디스토션 이펙터의 성격을 포착할 수 있습니다.
5. **스펙트럴 플랫니스 (Spectral Flatness)**: 스펙트럼이 얼마나 평평한지를 측정하여, 특히 리버브와 같이 주파수 응답을 균일하게 하는 이펙터의 특성을 반영할 수 있습니다.
6. **스펙트럴 컨트라스트 (Spectral Contrast)**: 스펙트럼의 피크와 밸리 사이의 대비를 측정하여, 다양한 이펙터가 주파수 대역에 미치는 영향을 구별하는 데 도움을 줍니다.
7. **크로마 피처 (Chroma Feature)**: 음악의 조성을 반영하여, 특히 모듈레이션 이펙터의 변조 효과가 조성에 미치는 영향을 분석할 수 있습니다.

<aside>
💡 기타 원음과 이펙팅 사운드 분류 및 머신러닝 / cnn훈련 → GANsynth학습으로 effecting sound 생성

</aside>

# 목표

1. clean tone과 effecting sound 사이에 분류가 가능한 유의미한 data 찾기
2. 머신러닝을 통해 clean tone과 effecting sound 분류하기

# VERSION

## 1-1

- code
    
    ```python
    import os
    import pickle
    import librosa
    import soundfile as sf
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.impute import SimpleImputer
    
    #sound data 컴퓨터에서 불러오기, 각 파일 별 이펙터, 픽업 추출)
    def load_classify_audio_files(base_path):
    
      audio_data={}
      for root, dirs, files in os.walk(base_path):
        for file in files:
          if file.endswith((".wav", ".mp3")):
            path=os.path.join(root, file)
            parts=path.split(os.sep)
            file_name_no_ext  = os.path.splitext(parts[-1])[0]
            key=tuple(parts[-3:-1]+[file_name_no_ext])
            if key not in audio_data:
              audio_data[key]=[]
            data, sr = librosa.load(path, sr=None)
            audio_data[key].append((data, sr))
      return audio_data
    
    #sound data 특성 추출
    def ext_features(data, sr):
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=data, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=data))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sr), axis=1)
        chroma_feature = np.mean(librosa.feature.chroma_stft(y=data, sr=sr), axis=1)
    
        features = {
            'mfcc': mfcc,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            'spectral_flatness': spectral_flatness,
            'spectral_contrast': spectral_contrast,
            'chroma_feature': chroma_feature
        }
        return features
    
    #특성 저장
    features_file_path = 'features_by_key.pkl'
    
    user_input = input("Do you want to reset the data? (yes/no): ")
    
    if user_input.lower() == 'yes':
      if os.path.exists(features_file_path):
        os.remove(features_file_path)
        print("Data has been reset.")
      else:
        print("No data file found to reset.")
    
      base_path = '/Users/nyoung/Desktop/dev/project/1. guitar effecting sound encoder-decoder/sound samples'
      classified_audio_files = load_classify_audio_files(base_path)
    
      #파일 별 특성 배치
      features_by_key = {}
      for key, files in tqdm(classified_audio_files.items(), desc="Processing each effect configuration"):
        features_by_key[key]=[]
        for data, sr in tqdm(files, desc=f"Processing files for {key}", leave=False):
          features = ext_features(data, sr)
          features_by_key[key].append(features)
    
      with open('features_by_key.pkl', 'wb')as f:
        pickle.dump(features_by_key, f)
    
    #파일 불러오기
    else:
      if os.path.exists(features_file_path):
        with open(features_file_path, 'rb') as f:
          features_by_key = pickle.load(f)
        print("Data loaded from existing file.")
      else:
        print("No existing data file found. Please reset data to proceed.")
    
    #label 정의
    effect_mapping = {
        'BluesDriver': 'drive',
        'Chorus': 'chorus',
        'Clean': 'clean',
        'Digital Delay': 'delay',
        'Flanger': 'flanger',
        'Hall Reverb': 'reverb',
        'Phaser': 'phaser',
        'Plate Reverb': 'reverb',
        'RAT': 'drive',
        'Spring Reverb': 'reverb',
        'Sweep Echo': 'delay',
        'TapeEcho': 'delay',
        'TubeScreamer': 'drive'
    }
    
    label_mapping = {
        'clean': 0,
        'drive': 1,
        'reverb': 2,
        'delay': 3,
        'chorus': 4,
        'phaser': 5,
        'flanger': 6
    }
    
    features_list = []
    labels_list = []
    file_names = []
    
    for key, features in features_by_key.items():
      effect_type = key[0]
      category = effect_mapping.get(effect_type, None)
      if category:
        label = label_mapping[category]
        for feature in features:
          features_list.append(feature)
          labels_list.append(label)
          file_names.append(key[2])
    
    #example
    '''
    reverb_label = label_mapping['reverb']
    reverb_features = [features_list[i] for i, label in enumerate(labels_list) if label == reverb_label]
    
    print(f"Number of 'reverb' features : {len(reverb_features)}")
    for feature in reverb_features[:5]:
      print(feature)
    '''
    
    # DataFrame 생성
    df = pd.DataFrame(features_list, columns=[
        'MFCC', 'Spectral Centroid', 'Spectral Rolloff', 'Zero Crossing Rate', 'Spectral Flatness', 'Spectral Contrast', 'Chroma Feature'])
    df['Label'] = labels_list
    
    # 데이터 전처리: 스케일링 및 NaN 값 대체
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')
    
    X_imputed = imputer.fit_transform(df.drop('Label', axis=1))
    X_scaled = scaler.fit_transform(X_imputed)
    
    # 데이터를 학습용과 테스트용으로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Label'], test_size=0.2, random_state=42)
    
    # 랜덤 포레스트 분류기 모델 선택 및 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 모델 예측 및 평가
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    
    # SVM 모델 선택 및 학습
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    
    # SVM 모델 예측 및 평가
    svm_y_pred = svm_model.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, svm_y_pred))
    print(classification_report(y_test, svm_y_pred, target_names=label_mapping.keys()))
    
    # 랜덤 포레스트 하이퍼파라미터 튜닝
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # 최적 모델 예측 및 평가
    best_y_pred = best_model.predict(X_test)
    print("Best Model Accuracy:", accuracy_score(y_test, best_y_pred))
    print(classification_report(y_test, best_y_pred, target_names=label_mapping.keys()))
    
    '''
    # 그래프 그리기
    selected_xticks = ['1-0', '2-0', '3-0', '4-0', '5-0', '6-0']
    
    features = ['MFCC', 'Spectral Centroid', 'Spectral Rolloff', 'Zero Crossing Rate', 'Spectral Flatness', 'Spectral Contrast', 'Chroma Feature']
    plt.figure(figsize=(14, 30))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(7, 1, i)
        sns.lineplot(data=df, x='File Name', y=feature, hue='Effect Type', palette='viridis')
        plt.title(feature)
        plt.xlabel('File Name (String-Fret)')
        plt.ylabel(feature)
        plt.xticks(selected_xticks, rotation=90)
        plt.legend(title='Effect Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    
    plt.tight_layout()
    
    output_file_path = '/Users/nyoung/Desktop/dev/feature_plots.png'
    plt.savefig(output_file_path)
    
    plt.show()
    '''
    ```
    
- result
    
    ![feature_plots.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/feature_plots.png)
    
    ```
       Accuracy: 0.2246376811594203
       
               precision    recall  f1-score   support
    
       clean       0.00      0.00      0.00       146
       drive       0.00      0.00      0.00       415
      reverb       0.00      0.00      0.00       421
       delay       0.22      1.00      0.37       403
      chorus       0.00      0.00      0.00       149
      phaser       0.00      0.00      0.00       130
     flanger       0.00      0.00      0.00       130
    
    accuracy                           0.22      1794
    macro avg      0.03      0.14      0.05      1794
    weighted avg   0.05      0.22      0.08      1794
    ```
    
    (ㅠㅠ…)
    

처음 이 프로젝트를 시작할 때는 막연히 fourier transforming을 통해서 구분을 해보자 생각했으나, 머신러닝을 위해서는 구분이 가능한 유의미한 데이터가 있어야한다고 생각해서 음형이 가질 수 있는 7가지 특성(MFCC, ZCR, Spectral Centroid, Spectral Rolloff, Zero Crossing Rate, Spectral Flatness, Spectral Contrast, Chroma Feature)을 추출하여 구분하려했다. 그러나 각 특성으로 effecting sound를 구분하는데에는 문제가 있었던 듯 싶다. 그래프를 확인해봤을때, 몇몇 feature에서 몇몇 effects는 유사한 graph개형을 가지며, y축의 위상 차이로 서로를 구분할수도 있을 것 같으나, 공통적으로 chorus, flanger에서(4,6번) graph가 튀는 것을 볼 수 있다. (그렇다고 다른 effect가 안튀는 것도 아니다…ㅠㅠ)왜 그런가 생각을 해보면 chorus, flanger계열은 원음의 음정을 살짝 변형시켜 원음과 함께 재생시키는 형태인데 그러면서 frequency가 많이 섞인듯 하다. 이번엔 원래 생각대로 fft를 통해 차이점을 찾아보자

## 1-2-1

- code
    
    ```python
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import librosa
    import matplotlib.pyplot as plt
    from scipy.fft import fft
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from tqdm import tqdm
    
    #파일 불러오기 + 파일별 이펙터, 픽업 추출
    def load_classify_files(base_path):
      audio_data =  {}
      for root, dirs, files in os.walk(base_path):
        for file in files:
          if file.endswith((".wav", ".mp3")):
            path = os.path.join(root, file)
            parts = path.split(os.sep)
            file_no_ext = os.path.splitext(parts[-1])[0]
            key = tuple(parts[-3:-1]+[file_no_ext])
            if key not in audio_data:
              audio_data[key]=[]
            data, sr = librosa.load(path, sr=None)
            audio_data[key].append((data, sr))
      return audio_data
    
    #FFT feature 추출
    def ext_fft_features(data, sr, n_fft=2048):
      fft_vals = np.abs(fft(data, n=n_fft)[:n_fft//2])
      return fft_vals
    
    #feature 저장
    features_file_path = 'features_by_key.pkl'
    
    user_input = input("Do you want to reset the data? (yes/no): ")
    
    if user_input.lower()=='yes':
      if os.path.exists(features_file_path):
        os.remove(features_file_path)
        print("Data has been reset.")
      else:
        print("No data file found to reset.")
    
      base_path = '/Users/nyoung/Desktop/dev/project/1. guitar effecting sound encoder-decoder/sound samples'
      classified_audio_files = load_classify_files(base_path)
    
      #file별 feature 배치
      features_by_key = {}
      for key, files in tqdm(classified_audio_files.items(), desc="Processing each effect configuration"):
        features_by_key[key]=[]
        for data, sr in tqdm(files, desc=f"Processing files for {key}", leave=False):
          fft_features = ext_fft_features(data, sr)
          features_by_key[key].append(fft_features)
    
      with open(features_file_path, 'wb') as f:
        pickle.dump(features_by_key, f)
    
    # file 불러오기
    else:
        if os.path.exists(features_file_path):
            try:
                with open(features_file_path, 'rb') as f:
                    features_by_key = pickle.load(f)
                print("Data loaded from existing file.")
            except EOFError:
                print("Error: File is empty or corrupted. Please reset the data.")
                features_by_key = {}
        else:
            print("No existing data file found. Please reset data to proceed.")
            features_by_key = {}
    
    #data 준비
    features_list = []
    labels_list = []
    file_names  = []
    
    effect_mapping = {
      'BluesDriver': 'drive',
      'Chorus': 'chorus',
      'Clean': 'clean',
      'Digital Delay': 'delay',
      'Flanger': 'flanger',
      'Hall Reverb': 'reverb',
      'Phaser': 'phaser',
      'Plate Reverb': 'reverb',
      'RAT': 'drive',
      'Spring Reverb': 'reverb',
      'Sweep Echo': 'delay',
      'TapeEcho': 'delay',
      'TubeScreamer': 'drive'
    }
    
    label_mapping = {
      'clean': 0,
      'drive': 1,
      'reverb': 2,
      'delay': 3,
      'chorus': 4,
      'phaser': 5,
      'flanger': 6
    }
    
    for key, features in tqdm(features_by_key.items(), desc="Preparing data for modeling"):
      effect_type = key[0]
      category = effect_mapping.get(effect_type, None)
      if category in label_mapping:
        for feature in features:
          features_list.append(feature)
          labels_list.append(label_mapping[category])
          file_names.append(key[2])
    
    # graph 그리기
    '''
    output_dir = '/Users/nyoung/Desktop/dev/project/1. guitar effecting sound encoder-decoder/fft_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    def plot_fft_features(features_by_key, effect_mapping, output_dir):
        num_effects = len(effect_mapping) - list(effect_mapping.values()).count('ignore')
        fig, axs = plt.subplots(num_effects, 1, figsize=(15, 2 * num_effects))
        
        effect_index = 0
        for effect, category in effect_mapping.items():
            if category == 'ignore':
                continue
            for key, features in features_by_key.items():
                if key[0] == effect:
                    axs[effect_index].plot(features[0])
                    axs[effect_index].set_title(f'FFT Frequency Features for {effect}')
                    axs[effect_index].set_xlabel('Frequency Bin')
                    axs[effect_index].set_ylabel('Amplitude')
                    axs[effect_index].legend([effect])
                    break
            effect_index += 1
    
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_effects_fft_plot.png'))
        plt.show()
    
    plot_fft_features(features_by_key, effect_mapping, output_dir)
    '''
    
    #Data frame 생성
    df = pd.DataFrame(features_list)
    df['Label'] = labels_list
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('Label', axis=1))
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Label'], test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    ```
    
- result
    
    ![all_effects_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/all_effects_fft_plot.png)
    
    ![Clean_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Clean_fft_plot.png)
    
    ![BluesDriver_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/BluesDriver_fft_plot.png)
    
    ![RAT_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/RAT_fft_plot.png)
    
    ![TubeScreamer_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/TubeScreamer_fft_plot.png)
    
    ![Digital Delay_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Digital_Delay_fft_plot.png)
    
    ![Sweep Echo_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Sweep_Echo_fft_plot.png)
    
    ![TapeEcho_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/TapeEcho_fft_plot.png)
    
    ![Hall Reverb_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Hall_Reverb_fft_plot.png)
    
    ![Plate Reverb_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Plate_Reverb_fft_plot.png)
    
    ![Spring Reverb_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Spring_Reverb_fft_plot.png)
    
    ![Chorus_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Chorus_fft_plot.png)
    
    ![Phaser_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Phaser_fft_plot.png)
    
    ![Flanger_fft_plot.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Flanger_fft_plot.png)
    
    ```
    Random Forest Accuracy: 0.6917502787068004
                precision    recall  f1-score   support
       clean       0.40      0.01      0.03       146
       drive       0.88      0.84      0.86       415
      reverb       0.79      0.98      0.87       421
       delay       0.57      0.97      0.72       403
      chorus       0.39      0.08      0.13       149
      phaser       0.49      0.18      0.27       130
     flanger       0.50      0.40      0.44       130
    
    accuracy                           0.69      1794
    macro avg      0.57      0.50      0.47      1794
    weighted avg   0.65      0.69      0.63      1794
    ```
    

조금은 정확도가 올랐다..! FFT의 frequency graph를 각 effector별로 한 개 씩 뽑아봤을때도 각 카테고리별로 유의미한 공통점을 확인할 수 있었다. 그러나 clean, modulation 카테고리가 전체적으로 정확도가 떨어지는 것을 확인할 수 있었는데(특히 recall값이 너무낮다…)그건 아마도 data개수가 drive, reverb, delay보다 3배 가까이 작기 때문에 그런 것 같다. 아마 데이터 증강이 조금 필요한 상황이 아닐까 생각해본다.

1. 괜찮은 data를 찾았다! 사실 지금 학습에 쓰이는 data의 수는 각 effetor마다 대략 100개 밖에 안되어서 학습을 할 때 어느정도 부족한 점이 있었는데, 이번 꺼는 각 effector마다 1000개씩 있다 종류도 다양한듯(압축파일이 6.4G라 받는 시간이 엄청나게 걸리는것이 단점이라면 단점이다…) 이걸 추가로 학습해봐야할 듯)
    
    [IDMT-SMT-Audio-Effects Dataset](https://zenodo.org/records/7544032)
    
2. 추가적으로 시간에 따른 FFT data를 받을 수 있는지도 알아봐야할 것 같다. → STFT 써보기
    
    [STFT(Short-Time Fourier Transform)와 Spectrogram의 python구현과 의미](https://sanghyu.tistory.com/37)
    

## 1-2-2

새로운 데이터를 학습하는 과정에서 자꾸만 아래와 같은 오류가 나면서 kill 되었다…

```jsx
Processing each effect configuration:  91%|▉| 37657/41334 [04:
zsh: killed     /usr/bin/python3 7658):   0%| | 0/1 [00:00<?, ?i
(base) nyoung@iyun-yeong-ui-MacBookPro ~ % 
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: 
There appear to be 1 leaked semaphore objects to clean up at shutdown
warnings.warn('resource_tracker: There appear to be %d '
```

왤까….하며 인터넷에 찾아봤을 때에도 명쾌한 답변을 찾지 못했다…

![스크린샷 2024-05-22 오후 5.07.58.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-05-22_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_5.07.58.png)

~~(역시 형이야 구해주러 왔구나….)~~

내 친구 gpt는 

![스크린샷 2024-05-22 오후 5.09.20.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2024-05-22_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%2592%25E1%2585%25AE_5.09.20.png)

라고 해서 일단 이렇게 해볼까 한다.

처리 완료 batch에 대해서 배웠다.→ 실패

- 학습 내용
    
    ### **컴퓨터 메모리 구조**
    
    1. **주 메모리(RAM)**:
        - 주 메모리는 컴퓨터가 데이터를 처리하는 데 사용하는 임시 저장 공간입니다. 빠르게 접근할 수 있지만 용량이 제한되어 있습니다.
        - 프로그램이 실행될 때 필요한 데이터와 코드가 주 메모리에 로드됩니다.
    2. **가상 메모리**:
        - 주 메모리가 부족할 때, 하드 디스크의 일부를 메모리처럼 사용하는 것을 가상 메모리라고 합니다. 이 영역을 스왑 영역(swap space)이라고도 합니다.
        - 가상 메모리는 실제 RAM보다 접근 속도가 느리기 때문에, 가상 메모리를 과도하게 사용하면 성능이 저하됩니다.
    
    ### **배치 처리가 메모리를 효율적으로 사용하는 이유**
    
    1. **메모리 사용 감소**:
        - 한 번에 많은 양의 데이터를 처리하면, 해당 데이터를 모두 주 메모리에 로드해야 합니다. 데이터가 너무 많으면 주 메모리 용량을 초과하게 되어 가상 메모리를 사용하게 됩니다. 이는 성능 저하와 함께 시스템의 불안정을 초래할 수 있습니다.
        - 데이터를 작은 배치로 나누어 처리하면, 한 번에 처리해야 할 데이터의 양이 줄어들어 주 메모리의 사용량이 줄어듭니다. 이렇게 하면 주 메모리를 효율적으로 사용할 수 있습니다.
    2. **캐시 효율성**:
        - CPU에는 데이터를 빠르게 접근할 수 있도록 도와주는 캐시 메모리가 있습니다. 배치 처리를 통해 작은 데이터 세트를 반복적으로 처리하면, 해당 데이터가 CPU 캐시에 더 잘 적재될 수 있습니다. 이는 데이터 접근 속도를 높이고 전체 처리 속도를 향상시킵니다.
    3. **가비지 컬렉션 관리**:
        - 파이썬과 같은 언어에서는 자동으로 메모리를 관리하는 가비지 컬렉터가 있습니다. 배치 처리를 통해 메모리 사용량을 줄이면, 가비지 컬렉션이 더 효율적으로 동작할 수 있습니다. 이는 메모리 누수를 방지하고 프로그램의 안정성을 높이는 데 도움이 됩니다.
    
    ### **시스템 리소스 관리**
    
    1. **CPU 사용량**:
        - 한 번에 많은 데이터를 처리하려고 하면, CPU가 과부하에 걸릴 수 있습니다. 배치 처리는 작업을 작은 단위로 나누어 CPU가 과부하 없이 효율적으로 작업을 수행할 수 있도록 합니다.
    2. **디스크 I/O**:
        - 데이터를 로드하고 저장하는 과정에서 디스크 입출력이 발생합니다. 배치 처리를 통해 디스크 I/O를 분산시키면, 디스크의 병목 현상을 줄이고 전체적인 성능을 향상시킬 수 있습니다.
    3. **병렬 처리**:
        - 배치 처리는 병렬 처리를 쉽게 적용할 수 있도록 합니다. 여러 배치를 동시에 처리함으로써 전체 처리 시간을 단축할 수 있습니다. 이는 멀티코어 프로세서의 성능을 최대한 활용하는 데 유리합니다.
    
    ### **데이터 배치란 무엇인가요?**
    
    데이터 배치는 컴퓨터가 한 번에 처리하는 데이터의 작은 그룹을 말합니다. 모든 데이터를 한꺼번에 처리하지 않고, 여러 번에 나누어 처리하는 방식입니다.
    
    ### **비유로 설명하기**
    
    생각해보세요. 여러분이 많은 양의 책을 정리해야 한다고 가정해 봅시다. 만약 모든 책을 한꺼번에 들려고 한다면, 너무 무거워서 힘들 것입니다. 대신, 작은 상자에 몇 권씩 나누어 담아 옮긴다면 더 쉽고 효율적일 것입니다. 데이터 배치도 이와 비슷한 원리입니다. 한꺼번에 많은 데이터를 처리하는 대신, 작은 배치로 나누어 처리하는 것입니다.
    
    ### **데이터 배치의 장점**
    
    1. **메모리 사용 감소**:
        - 한 번에 처리하는 데이터의 양을 줄이기 때문에, 메모리를 더 효율적으로 사용할 수 있습니다.
    2. **속도 향상**:
        - 작은 배치로 데이터를 처리하면, 시스템 리소스를 더 잘 관리할 수 있어 전체 처리 속도가 향상될 수 있습니다.
    3. **에러 방지**:
        - 모든 데이터를 한꺼번에 처리하려고 하면 메모리가 부족해져 프로그램이 중단될 수 있습니다. 배치로 나누면 이러한 문제를 줄일 수 있습니다.
    
    ### **예시**
    
    예를 들어, 10,000개의 오디오 파일을 처리해야 한다고 가정합시다. 한꺼번에 모든 파일을 처리하려면 컴퓨터의 메모리가 부족할 수 있습니다. 대신, 한 번에 1,000개의 파일만 처리하는 방식으로 데이터를 배치할 수 있습니다. 이렇게 하면 10,000개의 파일을 10번에 나누어 처리하게 됩니다.
    

### 실패 이유

느그 친구 gpt가 추천해준 코드를 면밀히 살펴보니 메모리 사용을 최소화 시키기 위해 batch를 설정하고 하나의 batch 특성 추가가 끝나면 그 batch를 초기화 시켜 버렸다(..?) 도대체 왜 그런 미친 짓을 했는지는 모르겠지만, 아무튼 결국엔 똑같이 메모리 부족으로 실패했다

### 개선 사항

google colab은 colab에서 gpu를 할당해준다고 한다. 그래서 개발 환경을 visual studio에서 google colab으로 변경할 예정이다.

추가로 sample data와 train data 분류하는 코드를 새로 짜보자

## 1-2-3-1

- code
    
    ```python
    import os
    import random
    import librosa
    import pickle
    import pandas as pd
    import numpy as np
    import librosa.display
    import matplotlib.pyplot as plt
    from scipy.fft import fft
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    base_path = '/content/drive/MyDrive/project/project1/sample'
    features_file_path = '/content/drive/MyDrive/project/project1/sample/features_by_key.pkl'
    output_dir = '/content/drive/MyDrive/project/project1/sample/stft_plots'
    
    #load files
    def load_audio_files(base_path, target_sr=22050, max_length=5.0):
      audio_data = {}
      file_index = 0
      for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
        files.sort()
        for file in tqdm(files, desc="Processing files", leave=False):
          if file.endswith((".wav", ".mp3")):
            parts=root.split(os.sep)
            effect=parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
            key = (effect, file_index)
            file_index += 1
            if key not in audio_data:
              audio_data[key]=[]
            file_path = os.path.join(root, file)
            data, sr = librosa.load(file_path, sr=None)
            #Amplitude Normalization
            data = librosa.util.normalize(data)
    
            audio_data[key].append((data, sr))
      return audio_data
    
    def load_sample_files(base_path, target_sr=22050, max_length=5.0):
        audio_data = {}
        file_index = 0
        for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
            effect_files = [file for file in files if file.endswith((".wav", ".mp3"))]
            if effect_files:
                file = random.choice(effect_files)  # 각 폴더에서 랜덤으로 1개의 파일 선택
                parts = root.split(os.sep)
                effect = parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
                key = (effect, file_index)
                file_index += 1
                if key not in audio_data:
                    audio_data[key] = []
                file_path = os.path.join(root, file)
                data, sr = librosa.load(file_path, sr=None)
                # Amplitude Normalization
                data = librosa.util.normalize(data)
                audio_data[key].append((data, sr))
        return audio_data
    
    #STFT funtion
    def ext_STFT_features(data, sr, n_fft=2048, hop_length=512):
      stft_vals = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
       return np.abs(stft_vals)
    
    # preparing data
    def data(audio_files, batch_size=100):
      features_list=[]
      labels_list=[]
      effect_mapping={
            'BluesDriver': 'drive',
            'Chorus': 'chorus',
            'Clean': 'clean',
            'Digital Delay': 'delay',
            'Distortion': 'drive',
            'FeedbackDelay': 'delay',
            'Flanger': 'flanger',
            'Hall Reverb': 'reverb',
            'NoFX': 'clean',
            'Overdrive': 'drive',
            'Phaser': 'phaser',
            'Plate Reverb': 'reverb',
            'RAT': 'drive',
            'SlapbackDelay': 'delay',
            'Spring Reverb': 'reverb',
            'Sweep Echo': 'delay',
            'TapeEcho': 'delay',
            'Tremolo': 'tremolo',
            'TubeScreamer': 'drive',
            'Vibrato': 'vibrato'
      }
    
      label_mapping={
          'clean' : 0,
          'drive' : 1,
          'reverb' : 2,
          'delay' : 3,
          'chorus' : 4,
          'phaser' : 5,
          'flanger' : 6,
          'tremolo' : 7,
          'vibrato' : 8
      }
    
      keys = list(audio_files.keys())
      for start in tqdm(range(0, len(keys), batch_size), desc='Preparing data in batches'):
        batch_keys = keys[start:start + batch_size]
        for key in batch_keys:
          effect_type = key[0]
          category = effect_mapping.get(effect_type, None)
          if category in label_mapping:
            for data, sr in audio_files[key]:
              fft_features = ext_STFT_features(data, sr)
              features_list.append((fft_features,  sr, effect_type))
              labels_list.append(label_mapping[category])
      return features_list, labels_list
    
    #plot spectogram graph
    def plot_spectogram(feature, sr, title):
      plt.figure(figsize=(10,4))
      librosa.display.specshow(librosa.amplitude_to_db(feature, ref=np.max), sr=sr, hop_length=512, x_axis='time', y_axis='log')
      plt.colorbar(format='%+2.0f dB')
      plt.title(title)
      plt.tight_layout()
      plt.show()
    
    audio_files = load_sample_files(base_path)
    features, labels = data(audio_files)
    
    for feature, sr, effect_type in features:
        title = f"Spectogram of {effect_type}"
        plot_spectogram(feature, sr, title)
    ```
    
- result
    
    ![Unknown-3.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-3.png)
    
    ![Unknown-4.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-4.png)
    
    ![Unknown-5.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-5.png)
    
    ![Unknown-6.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-6.png)
    
    ![Unknown-7.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-7.png)
    
    ![Unknown-8.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-8.png)
    
    ![Unknown-9.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-9.png)
    
    ![Unknown-10.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-10.png)
    
    ![Unknown-11.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-11.png)
    
    ![Unknown-12.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-12.png)
    
    ![Unknown-13.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-13.png)
    
    ![Unknown-14.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-14.png)
    
    ![Unknown-15.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-15.png)
    
    ![Unknown-16.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-16.png)
    
    ![Unknown-17.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-17.png)
    
    ![Unknown-18.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-18.png)
    
    ![Unknown-19.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-19.png)
    
    ![Unknown-20.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-20.png)
    
    ![Unknown-21.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-21.png)
    
    ![Unknown-22.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-22.png)
    
    ![Unknown-23.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-23.png)
    
    ![Unknown-24.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-24.png)
    
    ![Unknown-25.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-25.png)
    
    ![Unknown-26.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-26.png)
    
    ![Unknown-27.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-27.png)
    
    ![Unknown-28.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-28.png)
    
    ![Unknown-29.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-29.png)
    
    ![Unknown-30.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-30.png)
    
    ![Unknown-31.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-31.png)
    
    ![Unknown-32.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-32.png)
    
    ![Unknown-33.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-33.png)
    
    ![Unknown-34.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-34.png)
    
    ![Unknown-35.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-35.png)
    
    ![Unknown-36.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-36.png)
    
    ![Unknown-37.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-37.png)
    
    ![Unknown-38.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-38.png)
    
    ![Unknown-39.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-39.png)
    
    ![Unknown-40.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-40.png)
    
    ![Unknown-41.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-41.png)
    
    ![Unknown-42.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-42.png)
    
    ![Unknown-43.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-43.png)
    
    ![Unknown-44.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-44.png)
    
    ![Unknown-45.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-45.png)
    
    ![Unknown-46.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-46.png)
    
    ![Unknown-47.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-47.png)
    
    ![Unknown-48.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-48.png)
    
    ![Unknown-49.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-49.png)
    
    ![Unknown-50.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-50.png)
    
    ![Unknown-51.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-51.png)
    
    ![Unknown-52.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-52.png)
    
    ![Unknown-53.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-53.png)
    
    ![Unknown-54.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-54.png)
    
    ![Unknown-55.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-55.png)
    
    ![Unknown-56.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-56.png)
    
    ![Unknown-57.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-57.png)
    
- 수정사항
    
    ### 1. 개발 환경
    
    visual studio → google colab
    
    colab이 ram지원을 해줌으로써 컴퓨터 메모리으로만 돌릴 때는 kill됐던 데이터 로딩이 이제는 가능해지게 되었다.
    
    데이터 파일도 구글 드라이브에 넣어서 연동시킬 수 있기 때문에 컴퓨터 용량은 더 챙길 수 있는 것은 덤
    
    ### 2. sample 선택
    
    load_sample_files 함수 도입
    
    메모리 사용량을 줄이기 위해 처음에는 batch를 사용하는 등의 방법을 선택했으나 소요시간은 비슷했음 어짜피 지금은 데이터 feature가 잘 나오는지를 확인하고 싶은거라 각 이펙터별(폴더별) 샘플 데이터 하나만 선택하는 함수를 새로 제작 → 시간을 획기적으로 줄임
    

결과를 보아하니 각 이펙터별 특성이 잘 드러나있는 것 같다. 그리고 특히 1번 data가 확실히 data 수는 적어도 퀄리티는 2번보다 훨 좋은듯

data 중에 약 0.5초 delay된 data들이 많은데 이를 어떻게 해결해야할지 고민해봐야겠다

## 1-2-3-2

- code
    
    ```python
    import os
    import random
    import librosa
    import pickle
    import pandas as pd
    import numpy as np
    import librosa.display
    import matplotlib.pyplot as plt
    from scipy.fft import fft
    from sklearn.model_selection import train_test_split
    from tqdm.auto import tqdm
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    base_path = '/content/drive/MyDrive/project/project1/sample'
    features_file_path = '/content/drive/MyDrive/project/project1/sample/features_by_key.pkl'
    output_dir = '/content/drive/MyDrive/project/project1/sample/stft_plots'
    #remove silent segment
    def remove_silent(data, sr, db_threshold=-20, freq_threshold=4096, hop_length=512, n_fft=2048):
    
      stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
      freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
      freq_idx = np.where(freqs >= freq_threshold)[0][0]
    
      magnitude = np.abs(stft[freq_idx:, :])
      db_values = librosa.amplitude_to_db(magnitude, ref=np.max)
    
      start_frame = np.argmax(np.any(db_values > db_threshold, axis=0))
    
      if start_frame == 0 and np.all(db_values[:,0]<=db_threshold):
        return stft
    
      return stft[:, start_frame:]
    
    #load files
    def load_audio_files(base_path, target_sr=22050, max_length=5.0):
      audio_data = {}
      file_index = 0
      for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
        files.sort()
        for file in tqdm(files, desc="Processing files", leave=False):
          if file.endswith((".wav", ".mp3")):
            parts=root.split(os.sep)
            effect=parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
            key = (effect, file_index)
            file_index += 1
            if key not in audio_data:
              audio_data[key]=[]
            file_path = os.path.join(root, file)
            data, sr = librosa.load(file_path, sr=None)
            #Trim leading and trailing silence
            data, _ = librosa.effects.trim(data)
    
            #Remove silent segment
            data = remove_silent(data, sr, db_threshold=-20, freq_threshold=4096)
            
            # Amplitude Normalization
            data = librosa.util.normalize(data)
    
            audio_data[key].append((data, sr))
      return audio_data
    
    def load_sample_files(base_path, target_sr=22050, max_length=5.0):
        audio_data = {}
        file_index = 0
        for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
            effect_files = [file for file in files if file.endswith((".wav", ".mp3"))]
            if effect_files:
                file = random.choice(effect_files)  # 각 폴더에서 랜덤으로 1개의 파일 선택
                parts = root.split(os.sep)
                effect = parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
                key = (effect, file_index)
                file_index += 1
                if key not in audio_data:
                    audio_data[key] = []
                file_path = os.path.join(root, file)
                data, sr = librosa.load(file_path, sr=None)
                #Trim leading and trailing silence
                data, _ = librosa.effects.trim(data)
    
                #Remove silent segment
                data = remove_silent(data, sr, db_threshold=-20, freq_threshold=4096)
    
                # Amplitude Normalization
                data = librosa.util.normalize(data
                
                audio_data[key].append((data, sr))
        return audio_data
    
    #STFT funtion
    '''
    def ext_STFT_features(data, sr, n_fft=2048, hop_length=512):
      stft_vals = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
       return np.abs(stft_vals)
    '''
    
    # preparing data
    def data(audio_files, batch_size=100):
      features_list=[]
      labels_list=[]
      effect_mapping={
            'BluesDriver': 'drive',
            'Chorus': 'chorus',
            'Clean': 'clean',
            'Digital Delay': 'delay',
            'Distortion': 'drive',
            'FeedbackDelay': 'delay',
            'Flanger': 'flanger',
            'Hall Reverb': 'reverb',
            'NoFX': 'clean',
            'Overdrive': 'drive',
            'Phaser': 'phaser',
            'Plate Reverb': 'reverb',
            'RAT': 'drive',
            'SlapbackDelay': 'delay',
            'Spring Reverb': 'reverb',
            'Sweep Echo': 'delay',
            'TapeEcho': 'delay',
            'Tremolo': 'tremolo',
            'TubeScreamer': 'drive',
            'Vibrato': 'vibrato'
      }
    
      label_mapping={
          'clean' : 0,
          'drive' : 1,
          'reverb' : 2,
          'delay' : 3,
          'chorus' : 4,
          'phaser' : 5,
          'flanger' : 6,
          'tremolo' : 7,
          'vibrato' : 8
      }
    
      keys = list(audio_files.keys())
      for start in tqdm(range(0, len(keys), batch_size), desc='Preparing data in batches'):
        batch_keys = keys[start:start + batch_size]
        for key in batch_keys:
          effect_type = key[0]
          category = effect_mapping.get(effect_type, None)
          if category in label_mapping:
            for stft, sr in audio_files[key]:
              features_list.append((stft,  sr, effect_type))
              labels_list.append(label_mapping[category])
      return features_list, labels_list
    
    #plot spectogram graph
    def plot_spectogram(feature, sr, title):
      plt.figure(figsize=(10,4))
      librosa.display.specshow(librosa.amplitude_to_db(feature, ref=np.max), sr=sr, hop_length=512, x_axis='time', y_axis='log')
      plt.colorbar(format='%+2.0f dB')
      plt.title(title)
      plt.tight_layout()
      plt.show()
    
    audio_files = load_sample_files(base_path)
    features, labels = data(audio_files)
    
    for feature, sr, effect_type in features:
        title = f"Spectogram of {effect_type}"
        plot_spectogram(feature, sr, title)
    ```
    
- result
    
    ![Unknown-59.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-59.png)
    
    ![Unknown-60.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-60.png)
    
    ![Unknown-61.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-61.png)
    
    ![Unknown-62.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-62.png)
    
    ![Unknown-63.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-63.png)
    
    ![Unknown-64.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-64.png)
    
    ![Unknown-65.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-65.png)
    
    ![Unknown-66.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-66.png)
    
    ![Unknown-67.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-67.png)
    
    ![Unknown-68.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-68.png)
    
    ![Unknown-69.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-69.png)
    
    ![Unknown-70.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-70.png)
    
    ![Unknown-71.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-71.png)
    
    ![Unknown-72.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-72.png)
    
    ![Unknown-73.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-73.png)
    
    ![Unknown-74.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-74.png)
    
    ![Unknown-75.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-75.png)
    
    ![Unknown-76.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-76.png)
    
    ![Unknown-77.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-77.png)
    
    ![Unknown-78.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-78.png)
    
    ![Unknown-79.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-79.png)
    
    ![Unknown-80.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-80.png)
    
    ![Unknown-81.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-81.png)
    
    ![Unknown-82.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-82.png)
    
    ![Unknown-83.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-83.png)
    
    ![Unknown-84.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-84.png)
    
    ![Unknown-85.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-85.png)
    
    ![Unknown-86.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-86.png)
    
    ![Unknown-87.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-87.png)
    
    ![Unknown-88.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-88.png)
    
    ![Unknown-89.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-89.png)
    
    ![Unknown-90.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-90.png)
    
    ![Unknown-91.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-91.png)
    
    ![Unknown-92.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-92.png)
    
    ![Unknown-93.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-93.png)
    
    ![Unknown-94.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-94.png)
    
    ![Unknown-95.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-95.png)
    
    ![Unknown-96.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-96.png)
    
    ![Unknown-97.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-97.png)
    
    ![Unknown-98.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-98.png)
    
    ![Unknown-99.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-99.png)
    
    ![Unknown-100.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-100.png)
    
    ![Unknown-101.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-101.png)
    
    ![Unknown-102.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-102.png)
    
    ![Unknown-103.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-103.png)
    
    ![Unknown-104.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-104.png)
    
    ![Unknown-105.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-105.png)
    
    ![Unknown-106.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-106.png)
    
    ![Unknown-107.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-107.png)
    
    ![Unknown-108.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-108.png)
    
    ![Unknown-109.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-109.png)
    
    ![Unknown-110.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-110.png)
    
    ![Unknown-111.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-111.png)
    
    ![Unknown-112.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-112.png)
    
    ![Unknown-113.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Unknown-113.png)
    
- 수정사항
    
    ### 1. remove_silent 함수 추가
    
    remove silent 함수에서 frequence, magnitude threshold를 설정하고, 처음으로 threshold가 넘는 구간이 나타나면 그 구간을 sample의 시작으로 두었다
    
    또한 freq, mag를 측정하기 위해서는 librosa.stft를 또 돌려야하기 때문에 ext_stft를 제거했다
    
    ### 2. load_files 함수 수정
    
    librosa에서 제공하는 trim을 한번 거치고, 그다음 remove_silent함수를 거친 후에 Amplify를 나중에 거쳤다. Amplify를 먼저 거치니 계속 sample 2가 freq, db가 증가하게 되면서 trimming이 효과적으로 적용되지 않았던것 같다
    

여러번의 시행착오 끝에 모든 데이터 앞의 노이즈를 제거한 데이터를 만들었다..! 오랜 시간 끝에 드디어 인공지능 학습이 가능하게 된 것이다 이제 머신러닝을 통해 각종 학습 데이터들을 추출하고, cnn학습에 들어가보도록 하자

→ stft 데이터는 2차원 matrix이기 때문에 1차원 matrix에 대한 학습만 가능한 deep learning으로는 학습이 불가 / 바로 cnn학습으로 들어가자 들어가기에 앞서 cnn에 대해 공부를 조금 해보도록 하자

[딥러닝 CNN, 개념만 이해하기](https://youtu.be/9Cu2UfNO-gw?si=EODUu130rvfiPKTf)

[딥러닝 CNN, 컨볼루션 신경망, 합성곱 신경망 개념 정리](https://youtu.be/ZZKnBpd1lR4?si=2OuPxSicM87ZnkgP)

[Neural networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=4e9PDMYtE7XEzr8N)

[파이토치 PyTorch](https://youtube.com/playlist?list=PL7ZVZgsnLwEEIC4-KQIchiPda_EjxX61r&si=jrawP7GMW8IrTUWy)

## 1-3-1

- code
    
    ```python
    import os
    import random
    import librosa
    import pickle
    import pandas as pd
    import numpy as np
    import librosa.display
    import matplotlib.pyplot as plt
    from scipy.fft import fft
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    base_path = '/content/drive/MyDrive/project/project1/sample'
    features_file_path = '/content/drive/MyDrive/project/project1/sample/features_by_key.pkl'
    output_dir = '/content/drive/MyDrive/project/project1/sample/stft_plots'
    
    #remove silent segment
    def remove_silent(data, sr, db_threshold=-20, freq_threshold=4096, hop_length=512, n_fft=2048):
    
      stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
      freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
      freq_idx = np.where(freqs >= freq_threshold)[0][0]
    
      magnitude = np.abs(stft[freq_idx:, :])
      db_values = librosa.amplitude_to_db(magnitude, ref=np.max)
    
      start_frame = np.argmax(np.any(db_values > db_threshold, axis=0))
    
      if start_frame == 0 and np.all(db_values[:,0]<=db_threshold):
        return stft
    
      return stft[:, start_frame:]
    
    #load files
    def load_audio_files(base_path, target_sr=22050, max_length=5.0):
      audio_data = {}
      file_index = 0
      for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
        files.sort()
        for file in tqdm(files, desc="Processing files", leave=False):
          if file.endswith((".wav", ".mp3")):
            parts=root.split(os.sep)
            effect=parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
            key = (effect, file_index)
            file_index += 1
            if key not in audio_data:
              audio_data[key]=[]
            file_path = os.path.join(root, file)
            data, sr = librosa.load(file_path, sr=None)
            #Trim leading and trailing silence
            data, _ = librosa.effects.trim(data)
    
            #Remove silent segment
            data = remove_silent(data, sr, db_threshold=-20, freq_threshold=4096)
    
            # Amplitude Normalization
            data = librosa.util.normalize(data)
    
            audio_data[key].append((data, sr))
      return audio_data
    
    def load_sample_files(base_path, target_sr=22050, max_length=5.0):
        audio_data = {}
        file_index = 0
        for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
            effect_files = [file for file in files if file.endswith((".wav", ".mp3"))]
            if effect_files:
                file = random.choice(effect_files)  # 각 폴더에서 랜덤으로 1개의 파일 선택
                parts = root.split(os.sep)
                effect = parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
                key = (effect, file_index)
                file_index += 1
                if key not in audio_data:
                    audio_data[key] = []
                file_path = os.path.join(root, file)
                data, sr = librosa.load(file_path, sr=None)
                #Trim leading and trailing silence
                data, _ = librosa.effects.trim(data)
    
                #Remove silent segment
                data = remove_silent(data, sr, db_threshold=-20, freq_threshold=4096)
    
                # Amplitude Normalization
                data = librosa.util.normalize(data)
    
                audio_data[key].append((data, sr))
        return audio_data
    
    #STFT funtion
    '''
    def ext_STFT_features(data, sr, n_fft=2048, hop_length=512):
      stft_vals = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
      return np.abs(stft_vals)
    '''
    
    # preparing data
    def data(audio_files, batch_size=100):
      features_list=[]
      labels_list=[]
      effect_mapping={
            'BluesDriver': 'drive',
            'Chorus': 'chorus',
            'Clean': 'clean',
            'Digital-Delay': 'delay',
            'Distortion': 'drive',
            'FeedbackDelay': 'delay',
            'Flanger': 'flanger',
            'Hall-Reverb': 'reverb',
            'NoFX': 'clean',
            'Overdrive': 'drive',
            'Phaser': 'phaser',
            'Plate-Reverb': 'reverb',
            'RAT': 'drive',
            'SlapbackDelay': 'delay',
            'Spring-Reverb': 'reverb',
            'Sweep-Echo': 'delay',
            'TapeEcho': 'delay',
            'Tremolo': 'tremolo',
            'TubeScreamer': 'drive',
            'Vibrato': 'vibrato'
      }
    
      label_mapping={
          'clean' : 0,
          'drive' : 1,
          'reverb' : 2,
          'delay' : 3,
          'chorus' : 4,
          'phaser' : 5,
          'flanger' : 6,
          'tremolo' : 7,
          'vibrato' : 8
      }
    
      keys = list(audio_files.keys())
      for start in tqdm(range(0, len(keys), batch_size), desc='Preparing data in batches'):
        batch_keys = keys[start:start + batch_size]
        for key in batch_keys:
          effect_type = key[0]
          category = effect_mapping.get(effect_type, None)
          if category in label_mapping:
            for stft in audio_files[key]:
              features_list.append(stft)
              labels_list.append(label_mapping[category])
      return features_list, labels_list, label_mapping
    
    #plot spectogram graph
    def plot_spectogram(feature, sr, title):
      plt.figure(figsize=(10,4))
      librosa.display.specshow(librosa.amplitude_to_db(feature, ref=np.max), sr=sr, hop_length=512, x_axis='time', y_axis='log')
      plt.colorbar(format='%+2.0f dB')
      plt.title(title)
      plt.tight_layout()
      plt.show()
    
    #CNN_dataset
    class AudioDataset(Dataset):
      def __init__(self, features, labels):
        self.features = [f[0] for f in features]
        self.labels = labels
    
        self.features = [self._pad_feature(f) for f in self.features]
    
      def __len__(self):
        return len(self.features)
    
      def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = np.expand_dims(feature, axis=0)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
      def _pad_feature(self, feature):
        max_shape = (1025,470)
        padded_feature = np.zeros(max_shape)
        padded_feature[:feature.shape[0], :feature.shape[1]]=feature
        return padded_feature
    
    #CNN_model
    class AudioCNN(nn.Module):
      def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*128*58, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 =nn.Linear(128, num_classes)
    
      def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    audio_files = load_audio_files(base_path)
    
    features, labels, label_mapping  = data(audio_files)
    
    #for feature, sr, effect_type in features:
    #    title = f"Spectogram of {effect_type}"
    #    plot_spectogram(feature, sr, title)
    
    #generate dataset / dataloader
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(label_mapping)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    
    for epoch in tqdm(range(num_epochs)):
      model.train()
      running_loss = 0.0
      for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      print(f'Epoch : {epoch+1}/{num_epochs}, Loss : {running_loss/len(train_loader)}')
    
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    
      print(f'Accuracy: {100*correct/total}%')
    
    ```
    
- result
    
    ![Untitled.png](PROJECT%201%203b3a6a3def584c049c5a32639cace89d/Untitled.png)
    
- 수정사항
    
    ### 1. data 함수 수정
    
    features 데이터 타입이 튜플 (sr, features, effector types)로 저장되어 있어서 matrix 계산을 처리하는 함수들이 먹히지 않았다. 그리고 아마 labels와 차원이 맞지 않아(labels는 1차원이었음) 모델 훈련에서 오류가 났던 것 같다. 이를 위해 data함수에서 features_list에 feature만 저장하도록 수정하였다. 추가로 label_mapping을 Data 함수 안에서 로컬로 정의했기 때문에 추후 뒤에 훈련을 위해 label_mapping을 불러와야 하는데 자꾸 label_mapping을 출력하면 빈 셋이 출력되어 뭐때매 그런건 줄 몰랐는데, label_mapping도 data함수의 return 값으로 설정했더니 아주 잘 되었다. 새로운걸 배웠다…
    

92.5%라는 정확도가 나왔다 이전에 비해서 확실히 좋아지긴 했지만 아직 한참 미달이다. 여러가지 확인사항들을 확인해보고 정확도를 더 높여보자 일단 재현율이랑 정밀도를 측정해보고, 각종 parameter들을 조정하면서 정확도를 높여보도록 하자

## 1-3-2

- code
    
    ```python
    import os
    import random
    import librosa
    import pickle
    import pandas as pd
    import numpy as np
    import librosa.display
    import matplotlib.pyplot as plt
    from scipy.fft import fft
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import StandardScaler
    from tqdm.auto import tqdm
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    
    from google.colab import drive
    drive.mount('/content/drive')
    
    base_path = '/content/drive/MyDrive/project/project1/sample'
    features_file_path = '/content/drive/MyDrive/project/project1/sample/features_by_key.pkl'
    output_dir = '/content/drive/MyDrive/project/project1/sample/stft_plots'
    
    #remove silent segment
    def remove_silent(data, sr, db_threshold=-20, freq_threshold=4096, hop_length=512, n_fft=2048):
    
      stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length)
      freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
      freq_idx = np.where(freqs >= freq_threshold)[0][0]
    
      magnitude = np.abs(stft[freq_idx:, :])
      db_values = librosa.amplitude_to_db(magnitude, ref=np.max)
    
      start_frame = np.argmax(np.any(db_values > db_threshold, axis=0))
    
      if start_frame == 0 and np.all(db_values[:,0]<=db_threshold):
        return stft
    
      return stft[:, start_frame:]
      
      #load files
    def load_audio_files(base_path, target_sr=22050, max_length=5.0):
      audio_data = {}
      file_index = 0
      for root, dirs, files in tqdm(os.walk(base_path), desc='Walking through directories'):
        files.sort()
        for file in tqdm(files, desc="Processing files", leave=False):
          if file.endswith((".wav", ".mp3")):
            parts=root.split(os.sep)
            effect=parts[-3] if 'samples' not in parts[-2].lower() else parts[-1]
            key = (effect, file_index)
            file_index += 1
            if key not in audio_data:
              audio_data[key]=[]
            file_path = os.path.join(root, file)
            data, sr = librosa.load(file_path, sr=None)
            #Trim leading and trailing silence
            data, _ = librosa.effects.trim(data)
    
            #Remove silent segment
            data = remove_silent(data, sr, db_threshold=-20, freq_threshold=4096)
    
            # Amplitude Normalization
            data = librosa.util.normalize(data)
    
            audio_data[key].append((data, sr))
      return audio_data
      
      # preparing data
    def data(audio_files, batch_size=100):
      features_list=[]
      labels_list=[]
      effect_mapping={
            'BluesDriver': 'drive',
            'Chorus': 'chorus',
            'Clean': 'clean',
            'Digital-Delay': 'delay',
            'Distortion': 'drive',
            'FeedbackDelay': 'delay',
            'Flanger': 'flanger',
            'Hall-Reverb': 'reverb',
            'NoFX': 'clean',
            'Overdrive': 'drive',
            'Phaser': 'phaser',
            'Plate-Reverb': 'reverb',
            'RAT': 'drive',
            'SlapbackDelay': 'delay',
            'Spring-Reverb': 'reverb',
            'Sweep-Echo': 'delay',
            'TapeEcho': 'delay',
            'Tremolo': 'tremolo',
            'TubeScreamer': 'drive',
            'Vibrato': 'vibrato'
      }
    
      label_mapping={
          'clean' : 0,
          'drive' : 1,
          'reverb' : 2,
          'delay' : 3,
          'chorus' : 4,
          'phaser' : 5,
          'flanger' : 6,
          'tremolo' : 7,
          'vibrato' : 8
      }
    
      keys = list(audio_files.keys())
      for start in tqdm(range(0, len(keys), batch_size), desc='Preparing data in batches'):
        batch_keys = keys[start:start + batch_size]
        for key in batch_keys:
          effect_type = key[0]
          category = effect_mapping.get(effect_type, None)
          if category in label_mapping:
            for stft in audio_files[key]:
              features_list.append(stft)
              labels_list.append(label_mapping[category])
      return features_list, labels_list, label_mapping
      
    audio_files = load_audio_files(base_path)
    features, labels, label_mapping  = data(audio_files)
      
      #CNN_dataset
    class AudioDataset(Dataset):
      def __init__(self, features, labels):
        self.features = [f[0] for f in features]
        self.labels = labels
    
        self.features = [self._pad_feature(f) for f in self.features]
    
      def __len__(self):
        return len(self.features)
    
      def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        feature = np.expand_dims(feature, axis=0)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
      def _pad_feature(self, feature):
        max_shape = (1025,470)
        padded_feature = np.zeros(max_shape)
        padded_feature[:feature.shape[0], :feature.shape[1]]=feature
        return padded_feature
    
    #CNN_model
    class AudioCNN(nn.Module):
      def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*128*58, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 =nn.Linear(128, num_classes)
    
      def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
    #generate dataset / dataloader
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(label_mapping)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    
    for epoch in tqdm(range(num_epochs)):
      model.train()
      running_loss = 0.0
      for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
      print(f'Epoch : {epoch+1}/{num_epochs}, Loss : {running_loss/len(train_loader)}')
    
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
        for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
    
      print(f'Accuracy: {100*correct/total}%')
      
    from sklearn.metrics import precision_recall_fscore_support
    
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data,1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
    print(f"Precision: {precision: .4f}, Recall: {recall: .4f}, F1 Score: {f1: .4f}")
    ```
    
- result
- 수정사항

아직 못 돌려봄

GAN을 학습해보자

[[Pytorch] GAN 구현 및 학습](https://ddongwon.tistory.com/124)

[GANs vs. Diffusion Models: Putting AI to the test](https://aurorasolar.com/blog/putting-ai-to-the-test-generative-adversarial-networks-vs-diffusion-models/)

[[논문리뷰] GANSynth: Adversarial Neural Audio Synthesis (ICLR19)](https://music-audio-ai.tistory.com/13)

## 2-1-1

- code
    
    ```python
    #이하 생략
      
    audio_files = load_audio_files(base_path)
    features, labels, label_mapping  = data(audio_files)
      
    #data preparing for GAN generate
    clean_files = []
    drive_files = []
    
    for key, stfts in audio_files.items():
      effect_type=key[0]
      if effect_type in ['Clean','NoFX']:
        clean_files.extend([stft for stft in stfts])
      elif effect_type in ['BluesDriver', 'Distortion', 'Overdrive', 'RAT', 'TubeScreamer']:
        drive_files.extend([stft for stft in stfts])
    
    paired_data = []
    for clean, effect in zip(clean_files, drive_files):
      clean_stft, sr = clean
      effect_stft, _ = effect
    #  if clean_sr != effect_sr:
    #    raise ValueError('Sample rates do not match!')
      paired_data.append((clean_stft, effect_stft))
    
    class GANAudioDataset(Dataset):
      def __init__(self, paired_data):
        self.clean_stfts = [data[0] for data in paired_data]
        self.effect_stfts = [data[1] for data in paired_data]
        self.max_shape = (1025, max([stft.shape[1] for stft in self.clean_stfts+self.effect_stfts]))
    
      def __len__(self):
        return len(self.clean_stfts)
    
      def __getitem__(self, idx):
        clean_stft = self._pad_feature(self.clean_stfts[idx])
        effect_stft = self._pad_feature(self.effect_stfts[idx])
        clean_stft = np.stack((clean_stft.real, clean_stft.imag), axis=0)  
        effect_stft = np.stack((effect_stft.real, effect_stft.imag), axis=0)
        return torch.tensor(clean_stft, dtype=torch.float32), torch.tensor(effect_stft, dtype=torch.float32)
    
      def _pad_feature(self, feature):
        padded_feature = np.zeros(self.max_shape, dtype=np.complex64)
        padded_feature[:feature.shape[0], :feature.shape[1]]=feature
        return padded_feature
    
    dataset = GANAudioDataset(paired_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    class Generator(nn.Module):
      def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
      def forward(self, x):
        return self.main(x)
    
    class Discriminator(nn.Module):
      def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256,1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
      def forward(self, x):
        return self.main(x)
        
    #training model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))
    
    num_epochs = 50
    
    for epoch in tqdm(range(num_epochs), desc=f'{epoch+1} time updating'):
      for clean_stft, effect_stft in tqdm(dataloader):
        clean_stft, effect_stft = clean_stft.to(device), effect_stft.to(device)
        
        # Train Discriminator
        discriminator.zero_grad()
        real_outputs = discriminator(effect_stft)
        real_labels = torch.ones_like(real_outputs)
        real_loss = criterion(real_outputs, real_labels)
        real_loss.backward()
    
        fake_stft = generator(clean_stft)
        fake_outputs = discriminator(fake_stft.detach())
        fake_labels = torch.zeros_like(fake_outputs)
        fake_loss = criterion(fake_outputs, fake_labels)
        fake_loss.backward()
        optimizer_d.step()
    
        # Train Generator
        generator.zero_grad()
        fake_outputs = discriminator(fake_stft)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()
    
      print(f'Epoch [{epoch+1}/{num_epochs}], d_loss : {(real_loss.item()+fake_loss.item())/2}, g_loss : {g_loss.item()}')
    
    #Save the trained models
    generator_save_path = '/content/drive/MyDrive/project/project1/models/generator.pth'
    discriminator_save_path = '/content/drive/MyDrive/project/project1/models/discriminator.pth'
    
    # Save the trained models to Google Drive
    torch.save(generator.state_dict(), generator_save_path)
    torch.save(discriminator.state_dict(), discriminator_save_path)
    
    print(f"Generator model saved to {generator_save_path}")
    print(f"Discriminator model saved to {discriminator_save_path}")
    
    def load_generator(model_path, device):
        model = Generator().to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    # Function to convert STFT to time-domain signal
    def stft_to_audio(stft, sr, hop_length=512, n_fft=2048):
        stft_complex = stft[0] + 1j * stft[1]
        audio = librosa.istft(stft_complex, hop_length=hop_length)
        return audio
        
    def generate_effect_tone(clean_audio_path, generator, device, sr=22050, hop_length=512, n_fft=2048):
        clean_audio, _ = librosa.load(clean_audio_path, sr=sr)
        
        # Compute STFT
        clean_stft = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length)
        clean_stft = np.stack((clean_stft.real, clean_stft.imag), axis=0)
        clean_stft = torch.tensor(clean_stft, dtype=torch.float32).unsqueeze(0).to(device)
    
    generator_model_path = 'generator.pth'
    clean_audio_path = '/content/drive/MyDrive/project/project1/sample/sound samples 1/Clean/Clean/Bridge/1-13.wav'
    
    # Load the generator model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the generator model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = load_generator(generator_model_path, device)
    
    # Generate effect tone
    effect_audio = generate_effect_tone(clean_audio_path, generator, device)
    
    # Save the generated effect audio
    librosa.output.write_wav('generated_effect_audio.wav', effect_audio, sr=22050)
    ```
    
- result
- 수정사항

예상 런타임이 18시간이다…이거 어떻게 돌려야할지 좀 고민해봐야겠다…