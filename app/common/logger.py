import logging
import sys

# 配置日誌格式：時間 - 級別 - 模組 - 訊息
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout), # 輸出到控制檯
            # logging.FileHandler("app.log")   # 如果需要存到檔案可以開啟
        ]
    )

# 建立一個全域性的 logger 例項
logger = logging.getLogger("personal_chief")