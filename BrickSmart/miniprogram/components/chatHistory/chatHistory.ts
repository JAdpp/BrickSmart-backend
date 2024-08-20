Component({
  properties: {
    chatHistory: Boolean
  },
  data: {
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
  methods: {
    chatHis() {
      this.triggerEvent('lockWin', {chatHis: true})
      this.setData({
        chatHistory: false
      })
    },

    chatKey() {
      this.triggerEvent('lockWin', {chatHis: false})
      this.setData({
        chatHistory: true
      })
    }
  }
})