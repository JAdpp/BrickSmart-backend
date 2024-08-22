Page({
  data: {
    items: [],
  },

  onLoad() {
   
  },

  onShareAppMessage() {
    return {};
  },

  gotoPage:function(){
    wx.navigateTo({
      url:"../newPage/newPage"
    })
  },

  itemTap(e: any) {
    console.log(e.currentTarget.dataset.common)
    wx.navigateTo({
      url: '../construction/construction?info=' + JSON.stringify(e.currentTarget.dataset.common)
    })
  }
});