export HF_ENDPOINT=https://hf-mirror.com
# 定义一个函数，用于打开代理
proxy_on() {
  export HTTP_PROXY='http://sys-proxy-rd-relay.byted.org:8118'
  export http_proxy='http://sys-proxy-rd-relay.byted.org:8118'
  export https_proxy='http://sys-proxy-rd-relay.byted.org:8118'
  echo "Proxy is now ON"
}

# 定义一个函数，用于关闭代理
proxy_off() {
  unset HTTP_PROXY
  unset http_proxy
  unset https_proxy
  echo "Proxy is now OFF"
}