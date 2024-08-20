Page({
  data: {
    title: "big_house",
    img: "https://smart.gangcloud.cn/data/tutotialcover/cover1_1xzA6W7.png",
    num: 44,
    time: "2024-03-06 09:28:56"
  },

  onLoad(options: any) {
    if (options.info) {
      let info = JSON.parse(options.info)
      this.setData({
        title: info.title,
        img: info.img,
        num: info.num,
        time: info.time,
      })
    }
  },

  onShareAppMessage() {
    return {};
  },

  back2index() {
    wx.reLaunch({
      url: '../index/index'
    })
  },

  construction() {
    wx.navigateTo({
      url: '../construction/construction'
    })
  }
});