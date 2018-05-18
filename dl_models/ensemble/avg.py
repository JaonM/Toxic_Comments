import pandas as pd

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

clstm = pd.read_csv('../../submission/clstm_non_static_submit_lb_0.9792.csv', encoding='utf-8')[labels]
rnn = pd.read_csv('../../submission/text_rnn_non_static_submit.csv', encoding='utf-8')[labels]
fasttext = pd.read_csv('../../submission/fast_text_submit-lb-0.9773.csv', encoding='utf-8')[labels]
han = pd.read_csv('../../submission/han_non_static_submit.csv', encoding='utf-8')[labels]
cnn = pd.read_csv('../../submission/text_cnn_non_static_submit-lb-0.9755.csv',encoding='utf-8')[labels]

df_test = pd.read_csv('../../input/test.csv', encoding='utf-8')

submission = (clstm + rnn + fasttext + han+cnn) / 5
submission['id'] = df_test['id']
submission.to_csv('../../submission/clstm+han+rnn+fasttext+cnn_avg.csv', index=False, encoding='utf-8')
