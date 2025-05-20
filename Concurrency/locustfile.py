
from locust import HttpUser, task, between
import json

class BearerAuthUser(HttpUser):
    host = "https://xuanxiao.com:8888"  # 替换为你的服务地址
    wait_time = between(1, 3)  # 用户等待时间1-3秒

    def on_start(self):
        """用户启动时执行登录并获取token"""
        login_url = "/login"  # 替换为你的登录接口路径
        login_data = {
            "phone": "15016700370",  # 替换为实际用户名
            "password": "test123"   # 替换为实际密码
        }

        # 发送登录请求
        response = self.client.post(login_url, json=login_data)
        
        if response.status_code == 200:
            # 解析响应获取access_token
            token_data = json.loads(response.text)
            self.access_token = token_data.get("access_token")
        else:
            # 处理登录失败（这里简单抛出异常，实际可自定义处理）
            raise Exception("登录失败，状态码：%s" % response.status_code)

    @task(weight=1)
    def score(self):
        """请求分数"""
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        self.client.get("/score", headers=headers)  # 替换为你的API路径

    @task(weight=3)
    def chat(self):
        """对话"""
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        self.client.post("/bstreamagent", json={"dialog_id":"-1","content":"你好,浙江大学怎么样？","role":"用户"}, headers=headers) # 替换为你的API路径