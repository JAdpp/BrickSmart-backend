Page({
  data: {
    components: [],
    items: [
    ]
  },

  onLoad: function (options: any) {
    let this_ = this
    wx.request({
      url: "https://smart.gangcloud.cn/brick/getallcomponents/",
      method: "GET",
      success: function (res: any) {
        let components = []
        for (let i of res.data) {
          components.push({ id: i.sn, btype: i.btype, color: i.color, caption: i.caption, package: i.package, img: "https://smart.gangcloud.cn/" + i.imgFile, serialNumber: i.sn })
        }
        this_.setData({
          components: components
        })
      }
    })
    if (options.items) {
      let items = JSON.parse(options.items)
      this.setData({
        items: items
      })
    }
  },

  onShareAppMessage() {
    return {};
  },

  back() {
    let pages = getCurrentPages()
    let prevPage = pages[pages.length - 2]
    prevPage.setData({
      itmes: this.data.items
    });
    wx.navigateBack({
      delta: 1
    })
  },

  bindContentInput(e: any) {
    let currentTarget = e.currentTarget.dataset.item
    let itemList = this.data.items
    for (let item in itemList) {
      if (itemList[item].serialNumber == currentTarget.serialNumber) {
        if (e.detail.value == 0) {
          itemList.splice(item, 1)
        } else {
          itemList[item].num = e.detail.value
        }
        break
      }
    }
    this.setData({
      items: itemList
    })
  },

  insertCom(e: any) {
    let itemList = this.data.items
    let ifExisted = false
    for (let item in itemList) {
      if (itemList[item].serialNumber == e.currentTarget.dataset.common.serialNumber) {
        ifExisted = true
        break
      }
    }
    if (ifExisted == false) {
      console.log(e.currentTarget.dataset)
      this.setData({
        items: this.data.items.concat(
          { id: e.currentTarget.dataset.common.id, btype:  e.currentTarget.dataset.common.btype, color:  e.currentTarget.dataset.common.color, caption:  e.currentTarget.dataset.common.caption, package:  e.currentTarget.dataset.common.package, img:  e.currentTarget.dataset.common.img, serialNumber:  e.currentTarget.dataset.common.id, num: 1 }
        )
      })
    }
  },
});