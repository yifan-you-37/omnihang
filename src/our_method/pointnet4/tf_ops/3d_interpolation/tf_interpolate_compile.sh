# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
TF_INC=/scr-ssd/yifan/virtual_env/ilp-tf1/lib/python3.5/site-packages/tensorflow/include
TF_LIB=/scr-ssd/yifan/virtual_env/ilp-tf1/lib/python3.5/site-packages/tensorflow
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I$TF_INC -I /usr/local/cuda-9.0/include -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -lcudart -L /usr/local/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0
    