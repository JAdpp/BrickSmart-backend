Page({
  data: {
    setup: false
  },

  onShareAppMessage() {
    return {};
  },

  goBackPage: function(){
    wx.navigateBack()
  },
ToAdd:function(){
  wx.switchTab({
    url:'../customModule/customModule'
  })
},
  scan: function(){
    wx.navigateTo({
      url: '../scan/scan'
    })
  },
});