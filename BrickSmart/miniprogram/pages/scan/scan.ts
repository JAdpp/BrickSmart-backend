Page({
  data: {},

  onShareAppMessage() {
    return {};
  },

  back: function() {
    wx.navigateBack()
  },

  take: function(){
    const ctx = wx.createCameraContext()
    ctx.takePhoto({
      quality: 'high',
      success: (res) => {
        console.log(res.tempImagePath)
        wx. navigateBack()
      }
    })
  },
});