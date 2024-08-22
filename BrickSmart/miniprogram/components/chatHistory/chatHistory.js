// 引入插件
const plugin = requirePlugin('WechatSI');
// 获取全局唯一语音识别管理器
const manager = plugin.getRecordRecognitionManager();
Component({
  properties: {
    chatHistory: Boolean,
  },
  data: {
    recordState: 0, 
    textData:'',//语音转文字的文本内容
    content: '',//历史记录
    items: [
      {
        "content": "big_house",
        "createTime": "2024-03-06 09:28:56"
      },
      {
        "content": "12",
        "createTime": "2024-03-07 05:11:06"
      },
      {
        "content": "12",
        "createTime": "2024-03-12 12:41:40"
      }
    ],
  },
  lifetimes: {
    attached() {
      this.initSI();
    }
  },
  methods: {
     initSI() {
    const that = this;
    // 有新的识别内容返回，则会调用此事件
    manager.onRecognize = function (res) {
      console.log(res);
    };
    // 正常开始录音识别时会调用此事件
    manager.onStart = function (res) {
      console.log('成功开始录音识别', res);
      // 开始录音时-抖动一下手机
      wx.vibrateShort({ type: 'medium' });
    };
    // 识别错误事件
    manager.onError = function (res) {
      console.error('error msg', res);
      const tips = {
        '-30003': '说话时间间隔太短，无法识别语音',
        '-30004': '没有听清，请再说一次~',
        '-30011': '上个录音正在识别中，请稍后尝试',
      };
      const retcode = res?.retcode.toString();
      retcode &&
        wx.showToast({
          title: tips[`${retcode}`],
          icon: 'none',
          duration: 2000,
        });
    };
    //识别结束事件
    manager.onStop = function (res) {
      console.log('..............结束录音', res);
      console.log('录音临时文件地址 -->', res.tempFilePath);
      console.log('录音总时长 -->', res.duration, 'ms');
      console.log('文件大小 --> ', res.fileSize, 'B');
      console.log('语音内容 --> ', res.result);
      if (res.result === '') {
        wx.showModal({
          title: '提示',
          content: '没有听清，请再说一次~',
          showCancel: false,
        });
        return;
      }
      var text = that.data.content + res.result;
      that.setData({
        content: text,
      });
    };
  },
  // 手指触摸动作-开始录制语音
  touchStart() {
    this.setData({
      recordState: 1,
    });
    // 语音识别开始
    manager.start({
      duration: 30000,
      lang: 'zh_CN',
    });
  },
  // 手指触摸动作-结束录制
  touchEnd() {
    this.setData({
      recordState: 3,
    });
    // 语音识别结束
    manager.stop();
  },
    // initSI() {
    //   const that = this;
    //   // 有新的识别内容返回，则会调用此事件
    //   manager.onRecognize = (res: any) => {
    //     console.log(res);
    //   };
    //   // 正常开始录音识别时会调用此事件
    //   manager.onStart = (res: any) => {
    //     console.log('成功开始录音识别', res);
    //     // 开始录音时-抖动一下手机
    //     wx.vibrateShort({ type: 'medium' });
    //   };
    //   // 识别错误事件
    //   manager.onError = (res: any) => {
    //     console.error('error msg', res);
    //     const tips: { [key: string]: string } = {
    //       '-30003': '说话时间间隔太短，无法识别语音',
    //       '-30004': '没有听清，请再说一次~',
    //       '-30011': '上个录音正在识别中，请稍后尝试',
    //     };
    //     const retcode = res?.retcode?.toString();
    //     if (retcode) {
    //       wx.showToast({
    //         title: tips[retcode] || '未知错误',
    //         icon: 'none',
    //         duration: 2000,
    //       });
    //     }
    //   };
    //   // 识别结束事件
    //   manager.onStop = (res: any) => {
    //     console.log('..............结束录音', res);
    //     console.log('录音临时文件地址 -->', res.tempFilePath);
    //     console.log('录音总时长 -->', res.duration, 'ms');
    //     console.log('文件大小 --> ', res.fileSize, 'B');
    //     console.log('语音内容 --> ', res.result);
    //     if (res.result === '') {
    //       wx.showModal({
    //         title: '提示',
    //         textData: '没有听清，请再说一次~',
    //         showCancel: false,
    //       });
    //       return;
    //     }
    //     const text = that.data.textData + res.result;
    //     that.setData({
    //       textData: text,
    //     });
    //   };
    // },
  
    // // 手指触摸动作-开始录制语音
    // touchStart() {
    //   this.setData({
    //     recordState: 1,
    //   });
    //   // 语音识别开始
    //   manager.start({
    //     duration: 30000,
    //     lang: 'zh_CN',
    //   });
    // },
  
    // // 手指触摸动作-结束录制
    // touchEnd() {
    //   this.setData({
    //     recordState: 3,
    //   });
    //   // 语音识别结束
    //   manager.stop();
    // },
    // chatHis() {
    //   this.triggerEvent('lockWin', {chatHis: true})
    //   this.setData({
    //     chatHistory: false
    //   })
    // },

    // chatKey() {
    //   this.triggerEvent('lockWin', {chatHis: false})
    //   this.setData({
    //     chatHistory: true
    //   })
    // },
    // onReady() {
    //   this.initSI();
    // },
  }
})