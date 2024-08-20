Page({
  data: {
    modelUrl:'',
    imageUrl:'',
  },

  onShareAppMessage() {
    return {};
  },

  onLoad(options: Record<string, string | undefined>) {
    if (options.data) {
      try {
        let dataObject = JSON.parse(decodeURIComponent(options.data));
        // 接收并打印传递的参数
        console.log("模型URL: ", dataObject.modelUrl);
        console.log("图片URL: ", dataObject.imageUrl);

        // 设置数据到页面
        this.setData({
          modelUrl: dataObject.modelUrl,
          imageUrl: dataObject.imageUrl
        });
        wx.request({
          url: "https://smart.gangcloud.cn/brick/gettutorial/",
          method: "POST",
          data: {
            "userid": 123
          },
          header: {
            "Content-Type": 'application/json'
          },
          success: function (res) {
            console.log(res)
          }
        })
      } catch (error) {
        console.error("数据解析错误: ", error);
      }
    } else {
      console.error("没有传递数据");
    }
  },
  back() {
    wx.navigateBack()
  },
  
  
  setup() {
    wx.request({
      url: "https://smart.gangcloud.cn/brick/doprompt/",
      method: 'POST',
      data: {
        "userid": 123,
        "prompt": "1"
      },
      header: {
        "Content-Type": 'application/json'
      },
      success: function (res: any) {
        console.log(res.data)
      }
    })
    wx.navigateTo({
      url: '../setu/setu'
    })
  }

  // onShow() {
  //   setTimeout(()=>
  //   {
  //     console.log("hello World");
  //   }, 500)
  // }
});