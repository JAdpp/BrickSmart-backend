Page({
  data: {
    textData: '',
    ifDiagram: true,
    voiced:false,
    chatHis: false,
    chatHistory: false,
    lockWin: false,
    items: [null],
    added: false,
    referenceDiagram: '',
    modelUrl: '' ,  // 保存生成的模型URL，
    imageUrl:'',
    imageUrl1:'',
    loadingHidden: true,
  },

  onShow: function () {
    let pages = getCurrentPages();
    let currPage = pages[pages.length - 1];
    let info = currPage.data.itmes
    if (info) {
      this.setData({
        items: info,
        added: false,
      })
    }
  },

  onShareAppMessage() {
    return {};
  },

  back() {
    wx.switchTab({
      url:'../index/index',
      success: function(res) {
        // 页面跳转成功
        console.log('页面跳转成功');
      },
      fail: function(err) {
        // 页面跳转失败
        console.error('页面跳转失败:', err);
      }
    })
  },

  addDiagram() {
    if (this.data.textData && this.data.textData != '') {
      this.setData({
        ifDiagram: false,
        
        referenceDiagram: 'https://smart.gangcloud.cn/data/tutotialcover/cover1_1xzA6W7.png',
      })}
    //  else {
    //   wx.showModal({
    //     title: '提示',
    //     content: '请先填写想法',
    //     showCancel: false,
    //     success(res) {
    //       if (res.confirm) {
    //       }
    //     }
    //   })
    // }
  },
  Intalk(){
    wx.navigateTo({
      url: '../talk/talk',
      success: function(res) {
        // 页面跳转成功
        console.log('页面跳转成功');
      },
      fail: function(err) {
        // 页面跳转失败
        console.error('页面跳转失败:', err);
      }
    })
    
  },
  bindContentInput: function (e: { detail: { value: any; }; }) {
    this.setData({
      textData: e.detail.value,
      ifDiagram: false,
      added:true
    })
  },
  navigateDesignPage() {
    let modelUrl = 'https://www.freeimg.cn/thumbnails/5e8195e3e55fab4c472ce8f0d37eec31.png';
    let imageUrl = 'https://th.bing.com/th/id/OIP.rXcTjq9G8gH1w-4ThI3okgHaE8?w=245&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7';
    let dataObject = {
        modelUrl: modelUrl,
        imageUrl: imageUrl
    };
    wx.navigateTo({
        url: `/pages/design/design?data=${encodeURIComponent(JSON.stringify(dataObject))}`
    });
  },
   // 提交表单
   submitText() {
    if (!this.data.textData) {
      wx.showToast({
        title: '请输入提示信息',
        icon: 'none'
      });
      return;
    }
    else{
      this.setData({
        loadingHidden: false,//加载状态
      })
    }
    console.log("textData",this.data.textData)
    wx.request({
      url: 'http://127.0.0.1:8000/model/',  // Django 后端 API 地址
      method: 'POST',
      data: {
        prompt: this.data.textData  // 直接发送键值对，而不是 JSON 字符串
      },
      header: {
        'Content-Type': 'application/json'
      },
      success: (res) => {
        try {
          const data = typeof res.data === 'string' ? JSON.parse(res.data) : res.data;
          console.log("模型URL: " + data.model_filename + " 图片URL: " + data.image_filename);
          const abb=data.image_filename;
          this.setData({
            loadingHidden: false,
            
          })
          if (data && data.model_filename && data.image_filename) {
            this.setData({
              modelUrl: data.model_filename,
              imageUrl: data.image_filename,
            });
            let dataObject = {
              modelUrl: this.data.modelUrl,
              imageUrl: abb
          };
            // 自动跳转页面，带着自定义属性
            wx.navigateTo({
              url: `/pages/design/design?data=${encodeURIComponent(JSON.stringify(dataObject))}`
            });

            wx.showToast({
              title: '模型生成成功',
              icon: 'success'
            });
          } else {
            wx.showToast({
              title: '生成模型失败',
              icon: 'none'
            });
          }
        } catch (e) {
          wx.showToast({
            title: '响应解析失败',
            icon: 'none'
          });
        }
      },
      fail: (err) => {
        console.error(err);
        wx.showToast({
          title: '请求失败',
          icon: 'none'
        });
      }
    });
  },
  getFocus() {
    this.setData({
      chatHistory: true,
      chatHis: true,
      lockWin: false
    });
  },
  lockWin: function (e: any) {
    this.setData({
      lockWin: e.detail.chatHis
    })
  },

  loseFocus() {
    console.log(this.data.lockWin)
    if (this.data.lockWin == false) {
      this.setData({
        chatHis: false,
      })
    }
  },

  scan: function () {
    if (this.data.ifDiagram == false) {
      wx.navigateTo({
        url: '../scan/scan'
      })
    } else {
      wx.showModal({
        title: '提示',
        content: '请先填写想法或者生成参考图',
        showCancel: false,
        success(res) {
          if (res.confirm) {
          }
        }
      })
    }
  },

  editModule: function (e: any) {
    wx.navigateTo({
      url: '../customModule/customModule?items=' + JSON.stringify(e.currentTarget.dataset.items)
    })
  },

  custom: function () {
    if (this.data.ifDiagram == false) {
      wx.navigateTo({
        url: '../customModule/customModule'
      })
    } else {
      wx.showModal({
        title: '提示',
        content: '请先填写想法或者生成参考图',
        showCancel: false,
        success(res) {
          if (res.confirm) {
          }
        }
      })
    }
  },

  desgin() {
    // console.log(this.data.textData)
    if (this.data.added == true) {
      wx.navigateTo({
        url: '../design/design.wxml'
      })
    }
    wx.request({
      url:'http://127.0.0.1:3000/getdata',
      method:'POST',
      success(res){
        // const glbUrl = res.data[0].glb;
        console.log( res.data)
        // console.log(glbUrl);
      }
    })
    // wx.request({
    //   url:'http://127.0.0.1:3000/adddata',
    //   method:'POST',
    //   data:{
    //     valuetext:this.data.textData,
    //     glb:'https://7465-test-3-4gtecprf625fe63b-1325535039.tcb.qcloud.la/test/BoomBox.glb?sign=b13d705b9773923b55dc9ee82b1c85ae&t=1720094532'
    //   },
    //   success(res){
    //     console.log( res.data);
    //   }
    // })
  }
});