Page({
  data: {
    title: 'big_house',
    img: "https://smart.gangcloud.cn/data/tutotial/tuto1_YY7BoQl.png"
  },

  onLoad(options){
    if (options.info) {
      let info = JSON.parse(options.info)
      this.setData({
        title: info.name,
        img: "https://smart.gangcloud.cn" + info.imgFile,
      })
    }
  },

  onShareAppMessage() {
    return {};
  },

  back() {
    wx.reLaunch({
      url: '../index/index'
    })
  },
});