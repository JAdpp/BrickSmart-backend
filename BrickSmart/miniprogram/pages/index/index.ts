Page({
  data: {
    items: [],
  },

  onLoad() {
    let this_ = this
    wx.request({
      url: "https://smart.gangcloud.cn/brick/gettutorial/",
      method: 'POST',
      data: { "userid": "123" },
      header: {
        "Content-Type": 'application/json'
      },
      success: function (res: any) {
        console.log(res.data)
        let items = []
        let item
        for (let i of res.data) {
          item = { id: 0, img: "https://smart.gangcloud.cn/" + i.coverFile, name: i.title, totalNum: i.componentSum, createTime: i.createTime_Internal, imgFile: i.imgFile, description: i.description }
          items.push(item)
        }
        this_.setData({
          items:items
        })
      }
    })
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