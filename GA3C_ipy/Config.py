#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class Config:
    #게임 환경
    ATARI_GAME = 'PongDeterministic-v0'
    
    #학습된 에이전트 플레이 해보기
    PLAY_MODE = False
    
    #학습 모드
    TRAIN_MODELS = True
    
    #이전에 저장해둔 모델 불러오기
    LOAD_CHECKPOINT = False
    
    #0이면 마지막 체크포인트 불러짐
    LOAD_EPISODE = 0
    
    #agents 수
    AGENTS = 32
    
    #Predictors 수
    PREDICTORS = 2
    
    #TRAINERS 수
    TRAINERS = 2
    
    #GPU 사용
    DEVICE = 'gpu:0'
    
    #동적 조정
    DYNAMIC_SETTINGS = True
    DYNAMIC_SETTINGS_STEP_WAIT = 20
    DYNAMIC_SETTINGS_INITIAL_WAIT = 10
    
    #파라미터 설정
    DISCOUNT = 0.99
    TIME_MAX = 5
    
    #reward clipping
    REWARD_MIN = -1
    REWARD_MAX = 1
    
    #Queue 사이즈
    MAX_QUEUE_SIZE = 100
    PREDICTION_BATCH_SIZE = 128
    
    #DNN input
    STACKED_FRAMES = 4
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84
    
    #에피소드, 어닐링에피소드 수
    EPISODES = 400000
    ANNEALING_EPISODES_COUNT = 400000
    
    #엔트로피 정규화 파라미터
    BETA_START = 0.01
    BETA_END = 0.01
    
    #learing rate
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.0003
    
    #RMSProp parameters
    RMSPROP_DECAY = 0.99
    RAMSPROP_MOMENT = 0.0
    RAMSPROP_EPSILON = 0.1
    
    #Dual RMSProp - we found that using a single RMSProp for the two cost function works better and faster
    DUAL_RMSPROP = False
    
    #Gradient clipping
    USE_GRAD_CLIP = False
    GRAD_CLIP_NORM = 40.0
    
    #Epsilon (regularize policy lag in GA3C)
    LOG_EPSILON = 1e-6
    #TRAINING MIN BATCH SIZE
    TRAINING_MIN_BATCH_SIZE = 0
    
    ######################################
    #Log and Save
    
    TENSORBOARD = False
    
    #Tensorboard update steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000
    
    #모델 저장 유무
    SAVE_MODELS = True
    #저장 주기
    SAVE_FREQUENCY = 1000
    
    #stats 출력 주기
    PRINT_STATS_FREQUENCY = 1
    # ?
    STAT_ROLLING_MEAN_WINDOW = 1000
    #저장할 결과 파일명
    RESULTS_FILENAME = 'results.txt'
    #Network checkpoint name
    NATWORK_NAME = 'network'
    
    #########################################
    #추가적인 실험 파라미터
    MIN_POLICY = 0.0
    #log(softmax())대신 사용할 log_softmax()
    USE_LOG_SOFTMAX = False
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

