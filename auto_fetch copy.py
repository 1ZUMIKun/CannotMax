import csv
import datetime
from enum import Enum, auto
import logging
from pathlib import Path
import threading
import time
from typing import Literal
import cv2
import numpy as np
from sympy import N
import loadData
from recognize import MONSTER_COUNT, intelligent_workers_debug, RecognizeMonster
from predict import CannotModel
from collections.abc import Callable
from collections import deque

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

process_images = [cv2.imread(f"images/process/{i}.png") for i in range(16)]  # 16个模板

def match_images2(screenshot, templates):
    screenshot = cv2.resize(screenshot, (1920, 1080))
    confidence = float("-inf")
    loc = None
    best_id = -1
    for idx, template in enumerate(templates):
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > confidence:
            best_id, confidence, loc = idx, max_val, max_loc
    return best_id, confidence, loc

class Task:
    def __init__(self, name: str):
        self.name = name
        self.probe: Literal["template", "probefunc", ""] = "template" # 为空无条件满足
        self.probefunc: Callable[[], bool] = None
        self.probetmpl: list[cv2.typing.MatLike] = []
        self.next: list["Task"] = []
        self.errornext: list["Task"] = []
        self.action: Literal["clickpoint", "clicktmpl", "runfunc", ""] = "" # 为空不执行action
        self.clickpoint = (0, 0)
        self.runfunc: Callable[[], None] = None
        self.timeout = 120
        self.start_time = 0
        self.predelay = 0.0
        self.postdelay = 0.0
        self.roi = [(0, 0), (1, 1)]


class AutoFetch:
    def __init__(
        self,
        adb_connector: loadData.AdbConnector,
        game_mode,
        is_invest,
        update_prediction_callback: Callable[[float], None],
        update_monster_callback: Callable[[list], None],
        updater: Callable[[], None],
        start_callback: Callable[[], None],
        stop_callback: Callable[[], None],
        training_duration,
    ):
        self.adb_connector = adb_connector
        self.game_mode = game_mode  # 游戏模式（30人或自娱自乐）
        self.is_invest = is_invest  # 是否投资
        self.current_prediction = 0.5  # 当前预测结果，初始值为0.5
        self.recognize_results = []  # 识别结果列表
        self.incorrect_fill_count = 0  # 填写错误次数
        self.total_fill_count = 0  # 总填写次数
        self.update_prediction_callback = update_prediction_callback
        self.update_monster_callback = update_monster_callback
        self.updater = updater  # 更新统计信息的函数
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.image = None  # 当前图片
        self.image_name = ""  # 当前图片名称
        self.auto_fetch_running = False  # 自动获取数据的状态
        self.start_time = time.time()  # 记录开始时间
        self.training_duration = training_duration  # 训练时长
        self.data_folder = Path(f"data")  # 数据文件夹路径
        self.image_buffer = deque(maxlen=5)  # 图片缓存队列，设置队列长短来保存结算前的图片
        self.recognizer = RecognizeMonster()
        self.cannot_model = CannotModel()
        self.nowtask: Task = None
        self.nowtask = self.task_init()

    def task_init(self):
        start_task = Task("start_task")
        join_match = Task("join_match")
        select_single = Task("select_single")
        start_game = Task("start_game")
        recognize_predict = Task("recognize_predict")
        invest_left = Task("invest_left")
        invest_right = Task("invest_right")
        invest_skip = Task("invest_skip")
        wait_battle_result = Task("wait_battle_result")

        start_task.probe = ""
        start_task.next = [join_match]
        join_match.probetmpl = [cv2.imread(f"images/process/0_1.png")]
        join_match.action = "clicktmpl"
        # join_match.clickpoint = (0.9297, 0.8833) # 右ALL、返回主页、加入赛事、开始游戏
        join_match.next = [select_single]
        select_single.probetmpl = [cv2.imread(f"images/process/1_1.png")]
        select_single.action = "clicktmpl"
        # select_single.clickpoint = (0.8281, 0.8833) # 右礼物、自娱自乐
        select_single.next = [start_game]
        start_game.probetmpl = [cv2.imread(f"images/process/2_1.png")]
        start_game.action = "clicktmpl"
        start_game.predelay = 0.5
        start_game.next = [recognize_predict]
        # start_game.clickpoint = (0.9297, 0.8833) # 右ALL、返回主页、加入赛事、开始游戏
        recognize_predict.probetmpl = [cv2.imread(f"images/process/3_1.png")]
        recognize_predict.action = "runfunc"
        recognize_predict.runfunc = self.recognize_and_predict
        recognize_predict.next = [invest_skip]
        # invest_left.probe = "probefunc"
        # invest_left.probefunc = lambda: self.is_invest and self.current_prediction < 0.5
        # invest_left.action = "clickpoint"
        # invest_left.clickpoint = (0.8281, 0.8833) # 右礼物、自娱自乐
        # invest_left.next = [wait_battle_result]
        # invest_right.probe = "probefunc"
        # invest_right.probefunc = lambda: self.is_invest and self.current_prediction >= 0.5
        # invest_right.next = [wait_battle_result]
        invest_skip.probetmpl = [cv2.imread(f"images/process/3_1.png")]
        invest_skip.action = "clicktmpl"
        invest_skip.next = [wait_battle_result]
        wait_battle_result.probetmpl = [cv2.imread(f"images/process/8_1.png"), cv2.imread(f"images/process/9_1.png")]
        wait_battle_result.action = "runfunc"
        wait_battle_result.runfunc = lambda: 
        wait_battle_result.next = 


        return start_task


    def auto_fetch_run(self):
        if self.nowtask == None:
            logger.info("当前任务为空")
            return
        if len(self.image_buffer) == 0:
            logger.info("当前截图队列为空")
            return
        _, screenshot, _ = self.image_buffer[-1] # 获取最新图片
        for next_task in self.nowtask.next:
            match next_task.probe:
                case "template":
                    best_id, confidence, loc = match_images2(screenshot, next_task.probetmpl)
                    # logger.info(f"tmpl: {next_task.name}: {best_id} {confidence}, {loc}")
                    if best_id >= 0 and confidence > 0.8:
                        logger.info(f"Probe {next_task.name} by tmpl")
                        break
                case "":
                    break
                case _:
                    continue
        else:
            # 所有next_task均未满足
            logger.info("所有next_task均未满足，等待下一轮")
            return
        logger.info(f"Switch task {self.nowtask.name} to {next_task.name}")
        self.nowtask = next_task
        self.start_time = time.time()
        time.sleep(self.nowtask.predelay)
        match self.nowtask.action:
            case "clickpoint":
                logger.info(f"Click point {self.nowtask.clickpoint} by {self.nowtask.name}")
                self.adb_connector.click(next_task.clickpoint)
            case "clicktmpl":
                tmpl_w = self.nowtask.probetmpl[best_id].shape[1]
                tmpl_h = self.nowtask.probetmpl[best_id].shape[0]
                point = ((loc[0] + tmpl_w/2) / 1920, (loc[1] + tmpl_h/2) / 1080)
                logger.info(f"{self.nowtask.name}: Click template {point}")
                self.adb_connector.click(point)
            case "runfunc":
                next_task.runfunc()
            case _:
                pass
        time.sleep(next_task.postdelay)
        return

    def fill_data(self, battle_result, recoginze_results, image, image_name, result_image):
        # 获取队列头的图片
        if self.image_buffer:
            _, previous_image, _ = self.image_buffer[0]  # 获取队列头的图片
        else:
            logger.error("图片缓存队列为空，无法获取图片")
            previous_image = None

        if previous_image is None:
            logger.error("未找到1秒前的图片，无法保存")
            return
        image_data = np.zeros((1, MONSTER_COUNT * 2))

        for res in recoginze_results:
            region_id = res["region_id"]
            if "error" not in res:
                matched_id = res["matched_id"]
                number = res["number"]
                if matched_id != 0:
                    if region_id < 3:  # 左侧怪物
                        image_data[0][matched_id - 1] = number
                    else:  # 右侧怪物
                        image_data[0][matched_id + MONSTER_COUNT - 1] = number
            else:
                logger.error(f"存在错误，本次不填写")
                return

        image_data = np.append(image_data, battle_result)
        image_data = np.nan_to_num(image_data, nan=-1)  # 替换所有NaN为-1

        # 将数据转换为列表，并添加图片名称
        data_row = image_data.tolist()
        # 保存数据
        start_time = datetime.datetime.fromtimestamp(self.start_time).strftime(
            r"%Y_%m_%d__%H_%M_%S"
        )

        if intelligent_workers_debug:  # 如果处于debug模式，保存人工审核图片到本地
            data_row.append(image_name)
            if image is not None:
                image_path = self.data_folder / "images" / image_name
                cv2.imwrite(image_path, image)

            if previous_image is not None:
                image_path = self.data_folder / "images" / (image_name+"1s.png")
                cv2.imwrite(image_path, previous_image)
                logger.info(f"保存1秒前的图片到 {image_path}")

            # 新增保存结果图片逻辑
            # if self.image_name:
            #     result_image_name = self.image_name.replace(".png", "_result.png")
            #     # 缩放到128像素高度
            #     (h, w) = result_image.shape[:2]
            #     new_height = 128
            #     resized_image = cv2.resize(result_image, (int(w * (new_height / h)), new_height))
            #     image_path = self.data_folder / "images" / result_image_name
            #     cv2.imwrite(str(image_path), resized_image)
            #     logger.info(f"保存结果图片到 {image_path}")
        with open(self.data_folder / "arknights.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data_row)
        logger.info(f"写入csv完成")

    @staticmethod
    def calculate_average_yellow(image):
        def get_saturation(bgr):
            # 将BGR转换为0-1范围后计算饱和度
            b, g, r = [x / 255.0 for x in bgr]
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            return (delta / cmax) * 255 if cmax != 0 else 0  # 返回0-255范围的饱和度值

        if image is None:
            logger.error("图像加载失败")
            return None

        height, width, _ = image.shape

        # 获取左上角和右上角颜色
        left_top = image[0, 0]
        right_top = image[0, width - 1]  # 右上角坐标为(width-1, 0)

        # 计算饱和度
        sat_left = get_saturation(left_top)
        sat_right = get_saturation(right_top)

        # 计算饱和度差值
        saturation_diff = sat_left - sat_right

        # 检查差值是否符合要求，平局或者其他两边相等会被这个筛选掉
        if abs(saturation_diff) <= 20:
            logger.error(f"饱和度差值不足20 (左:{sat_left:.1f} vs 右:{sat_right:.1f})")
            return None

        # 返回左上角是否比右上角饱和度更高
        return saturation_diff > 20

    def save_recoginze_image(self, results, screenshot):
        """
        生成复核图片
        """
        x1 = int(0.2479 * self.adb_connector.screen_width)
        y1 = int(0.8444 * self.adb_connector.screen_height)
        x2 = int(0.7526 * self.adb_connector.screen_width)
        y2 = int(0.9491 * self.adb_connector.screen_height)
        # 截取指定区域
        roi = screenshot[y1:y2, x1:x2]
        # 处理结果
        processed_monster_ids = []  # 用于存储处理的怪物 ID
        for res in results:
            if "error" not in res:
                matched_id = res["matched_id"]
                if matched_id != 0:
                    processed_monster_ids.append(matched_id)  # 记录处理的怪物 ID
        # 生成唯一的文件名（使用时间戳）
        timestamp = int(time.time())
        # 将处理的怪物 ID 拼接到文件名中
        monster_ids_str = "_".join(map(str, processed_monster_ids))
        current_image_name = f"{timestamp}_{monster_ids_str}.png"
        current_image = cv2.resize(
            roi, (roi.shape[1] // 2, roi.shape[0] // 2)
        )  # 保存缩放后的图片到内存
        return current_image, current_image_name

    def save_statistics_to_log(self):
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, _ = divmod(remainder, 60)
        stats_text = (
            f"总共填写次数: {self.total_fill_count}\n"
            f"填写×次数: {self.incorrect_fill_count}\n"
            f"当次运行时长: {int(hours)}小时{int(minutes)}分钟\n"
        )
        with open("log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(stats_text)

    def recognize_and_predict(self, screenshot = None):
        if screenshot is None:
            screenshot = self.adb_connector.capture_screenshot()
        self.recognize_results = self.recognizer.process_regions(screenshot)
        # 获取预测结果
        self.update_monster_callback(self.recognize_results)
        left_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        right_counts = np.zeros(MONSTER_COUNT, dtype=np.int16)
        for res in self.recognize_results:
            if 'error' not in res:
                region_id = res['region_id']
                matched_id = res['matched_id']
                number = res['number']
                if matched_id == 0:
                    continue
                if region_id < 3:
                    left_counts[matched_id -1] = number
                else:
                    right_counts[matched_id -1] = number
            else:
                logger.error("识别结果有错误，本轮跳过")
        self.current_prediction = self.cannot_model.get_prediction(left_counts, right_counts)
        self.update_prediction_callback(self.current_prediction)
        # 人工审核保存测试用截图
        if intelligent_workers_debug:  # 如果处于debug模式且处于自动模式
            self.image, self.image_name = self.save_recoginze_image(
                self.recognize_results, screenshot
            )
            # ==============暂时保存图片全部================
            self.image=screenshot

    def battle_result(self, screenshot):
        # 判断本次是否填写错误，结果不等于None（不是平局或者其他）才能继续
        if self.calculate_average_yellow(screenshot) != None:
            if self.calculate_average_yellow(screenshot):
                self.fill_data(
                    "L", self.recognize_results, self.image, self.image_name, screenshot
                )
                if self.current_prediction > 0.5:
                    self.incorrect_fill_count += 1  # 更新填写×次数
                logger.info("填写数据左赢")
            else:
                self.fill_data(
                    "R", self.recognize_results, self.image, self.image_name, screenshot
                )
                if self.current_prediction < 0.5:
                    self.incorrect_fill_count += 1  # 更新填写×次数
                logger.info("填写数据右赢")
            self.total_fill_count += 1  # 更新总填写次数
            self.updater()  # 更新统计信息
            logger.info("下一轮")
            # 为填写数据操作设置冷却期
            # 平局或者其他也照常休息5秒


    def auto_fetch_loop(self):
        while self.auto_fetch_running:
            try:
                # self.auto_fetch_data()
                self.auto_fetch_run()
                elapsed_time = time.time() - self.start_time
                if self.training_duration != -1 and elapsed_time >= self.training_duration:
                    logger.info("已达到设定时长，结束自动获取")
                    break
                # 检测一次间隔时间——————————————————————————————————
                time.sleep(0.1)
            except Exception as e:
                logger.exception(f"自动获取数据出错:\n{e}")
                break
            # time.sleep(2)
        else:
            logger.info("auto_fetch_running is False, exiting loop")
            return
        # 不通过按钮结束自动获取
        logger.info("break auto_fetch_loop")
        self.stop_auto_fetch()

    def capture_loop(self):
        while self.auto_fetch_running:
            screenshot = self.adb_connector.capture_screenshot()
            if screenshot is None:
                logger.error("截图失败，无法继续操作")
                return
            # 保存当前截图及其信息到缓冲区
            timestamp = int(time.time())
            self.image_buffer.append((timestamp, screenshot.copy(), []))
            logger.info(f"向队列中添加截图，当前队列长度: {len(self.image_buffer)}")
            time.sleep(0.2)
        else:
            logger.info("auto_fetch_running is False, exiting capture loop")
            return

    def start_auto_fetch(self):
        if not self.auto_fetch_running:
            self.auto_fetch_running = True
            self.start_time = time.time()
            start_time = datetime.datetime.fromtimestamp(self.start_time).strftime(
                r"%Y_%m_%d__%H_%M_%S"
            )
            self.data_folder = Path(f"data/{start_time}")
            logger.info(f"创建文件夹: {self.data_folder}")
            self.data_folder.mkdir(parents=True, exist_ok=True)  # 创建文件夹
            (self.data_folder / "images").mkdir(parents=True, exist_ok=True)
            with open(self.data_folder / "arknights.csv", "w", newline="") as file:
                header = [f"{i+1}L" for i in range(MONSTER_COUNT)]
                header += [f"{i+1}R" for i in range(MONSTER_COUNT)]
                header += ["Result", "ImgPath"]
                writer = csv.writer(file)
                writer.writerow(header)
            self.log_file_handler = logging.FileHandler(
                self.data_folder / f"AutoFetch_{start_time}.log", "a", "utf-8"
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"
            )
            self.log_file_handler.setFormatter(file_formatter)
            self.log_file_handler.setLevel(logging.INFO)
            logging.getLogger().addHandler(self.log_file_handler)
            threading.Thread(target=self.auto_fetch_loop).start()
            threading.Thread(target=self.capture_loop).start()
            logger.info("自动获取数据已启动")
            self.start_callback()
        else:
            logger.warning("自动获取数据已在运行中，请勿重复启动。")

    def stop_auto_fetch(self):
        self.auto_fetch_running = False
        self.save_statistics_to_log()
        logger.info("停止自动获取")
        self.stop_callback()
        logging.getLogger().removeHandler(self.log_file_handler)
        # 结束自动获取数据的线程
