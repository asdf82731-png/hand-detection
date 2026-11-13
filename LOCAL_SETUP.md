# 地端執行方案

最後更新：2026-11-12

在你自己的電腦上完成「檢查資料 → 訓練 → 啟動辨識服務」。CSV、逐幀圖片、影片都留在本機，不需要上傳；要給 Claude 看的只有終端機印出的文字，以及幾張小的預覽圖。

> 2026-11-13 已在你的 Windows 電腦上完整跑過一次（Python 3.12、PyTorch 2.4.0 + CUDA 12.4、RTX 4060）。

## 1. 取得程式

需要先裝好 [Git](https://git-scm.com/) 和 Python 3.9–3.12（論文環境的 3.12 可以）。

```
git clone https://github.com/asdf82731-png/hand-detection.git
cd hand-detection
```

## 2. 安裝套件

如果論文時期的 Anaconda 環境還在（PyTorch 2.4.0 + GPU），可以直接啟用它，跳到「安裝其餘套件」。

否則建立新環境：

```
python -m venv .venv
.venv\Scripts\activate
```

先裝 GPU 版 PyTorch（RTX 4060 用 CUDA 12.4 版；如果先裝 `requirements.txt`，Windows 上會裝到只能用 CPU 的版本）：

```
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

安裝其餘套件：

```
pip install -r requirements.txt
```

確認 GPU 可用（應該印出 `True`）：

```
python -c "import torch; print(torch.cuda.is_available())"
```

印出 `False` 也能訓練，只是用 CPU，會慢很多（依資料量估計整個訓練要幾十分鐘）。

確認環境沒問題（應該顯示 `OK`）：

```
python -m unittest
```

## 3. 放資料

在 `hand-detection` 資料夾裡建立 `data` 資料夾（已設定不會被 git 上傳）。每個動作需要**一份標籤檔**加上**一份畫面來源**。

標籤檔請複製並改名：

| 原檔名 | 放到 |
|---|---|
| `Full_strike_processed.csv` | `data\Full.csv` |
| `down_stroke_data_1k.csv` | `data\Down.csv` |
| `Up_strike_processed.csv` | `data\Up.csv` |
| `Tab_processed.csv` | `data\Tab.csv` |

畫面來源二選一，**必須是當初標註時用的那一份**：

- **逐幀圖片（建議）**：把圖片放進 `data\Full\`、`data\Down\`、`data\Up\`、`data\Tab\`，檔名維持 `frame_0026.jpg` 這種格式。程式用檔名裡的數字對應 CSV 的 `frame_id`。
- **影片**：放成 `data\Full.mp4` 等。程式把影片第一幀當作第 0 幀；如果當初抽圖片的工具是從 `frame_0001` 開始編號，會差一幀，請改用逐幀圖片或告訴 Claude。

兩者都有時用圖片資料夾。缺標籤檔或缺畫面的動作會被跳過，其餘照常訓練；缺哪個動作，訓練出來的模型就不認得那個動作。

> 目前手上的四支「edit ver」影片各只有 466–503 幀，但 CSV 的幀編號到 1003–1499，兩者對不上，不能拿 edit ver 影片當畫面來源。

## 4. 檢查資料

```
python check_data.py --data-dir data
```

它不會改動任何資料，只會印出報告，並在 `data` 裡存下每個動作一張 `check_preview_<動作>.jpg`。逐項看：

1. **`frame numbers fit: OK`**：CSV 的幀編號落在畫面範圍內。出現 `PROBLEM` 代表這份 CSV 不是對這批畫面標的，要換成正確的畫面來源。
2. **`detected`**：兩隻手被偵測到的比例。edit ver 影片量到的是 84%–100%，明顯低於這個範圍就要留意。
3. **預覽圖**：
   - 人應該是正的（頭在上、打擊板在下）。如果是顛倒的，先停下來告訴 Claude。
   - 圖上兩個綠框分別標著 `slot left`、`slot right`，下方紅字是這一幀 CSV 的左右手標籤。看 `slot left` 框住的那隻手，在 CSV 裡是 left 還是 right。
4. **Summary 的 `--csv-left-slot`**：程式依「標成揮擊的幀，手應該動得比較多」推測的建議值。它只是參考，**以預覽圖為準**：`slot left` 框住的手在 CSV 裡叫 left，就用 `left`；叫 right，就用 `right`。

鏡頭正對鼓手時，畫面左邊的手是鼓手的右手。所以如果 CSV 的 left/right 指的是鼓手自己的左右手，答案通常是 `right`。

不確定的話，把整段輸出文字貼給 Claude，預覽圖也可以附上（每張不到 1MB）。

## 5. 訓練

```
python train.py --data-dir data --csv-left-slot <left 或 right>
```

每個 epoch 印一行 loss 和驗證準確率，結束時印出各類別的 precision / recall / F1、混淆矩陣、AUC-ROC，並存成 `multi_motion_model_7class.pt`（約 17MB）。

請把訓練的整段輸出文字貼給 Claude，用來判斷模型是否達標、要不要調整。

驗證集是每支影片時間軸的最後一段，跟訓練用的片段不重疊，所以數字會比論文的隨機切分低，但比較可信。

## 6. 啟動辨識服務

```
uvicorn main:app --port 8000
```

瀏覽器開啟 `http://localhost:8000`，上傳影片即可辨識。服務啟動時會讀取 `multi_motion_model_7class.pt`；還沒訓練就啟動會直接報錯。

## 7. 把模型帶回 GitHub（選用）

想讓雲端的 Claude 也能用這個模型繼續調整：

```
git add multi_motion_model_7class.pt
git commit -m "Add trained 7-class checkpoint"
git push
```

## 常見問題

| 狀況 | 處理 |
|---|---|
| `check_data.py` 顯示 `PROBLEM: the CSV labels frame ...` | 畫面來源不是標註時用的那一份，換成對的圖片或影片 |
| `MISSING label file` / `MISSING frames` | 檔名或位置不對，對照第 3 節 |
| 預覽圖是顛倒的 | 圖片當初抽出來時就是顛倒的，告訴 Claude，需要加一步轉正 |
| `torch.cuda.is_available()` 是 `False` | 重新執行第 2 節的 GPU 版 PyTorch 安裝；或先用 CPU 訓練 |
| `pip install` 找不到 `mediapipe==0.10.21` | Python 版本要在 3.9–3.12 之間 |
