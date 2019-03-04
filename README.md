# Chatbots kontekstual dengan Tensorflow
Dalam percakapan, konteks adalah raja! Kami akan membangun kerangka obrolan menggunakan 
Tensorflow dan menambahkan beberapa penanganan konteks untuk menunjukkan bagaimana ini bisa didekati.

### Prerequisites
Dalam menjalankan aplikasi memerlukan beberapa library tambahan:
```
nltk, flask, numpy,tensorflow
```

### Usage 

1. langsung run file ```appbot.py```
2. melkukan request ```curl -i -X POST -H "Content-Type: application/json" -d '{"me":"hai"}' http://localhost:9000/chat```
