# data_usage_demo 说明

这个文档配合 [data_usage_demo.py](/data/private/liliangchao/workspace/AD/examples/data_usage_demo.py) 使用。

它的目标不是训练模型，而是让你快速看懂当前 `data` 层会产出什么、每个字段代表什么，以及 `train / val / calibration / test` 这 4 个 split 在代码里是怎么表现的。

## 怎么运行

```bash
uv run python examples/data_usage_demo.py --category bottle
```

默认读取仓库里的：

```text
data/mvtec
```

输出是一段 JSON。

## 怎么读这段 JSON

建议按这个顺序看：

1. 先看顶层：`root`、`category`、`split_lengths`
2. 再看 `basic_demo.samples`
3. 最后看 `transform_demo`

这样最容易理解。

---

## 顶层字段

### `root`

```json
"root": "/data/private/liliangchao/workspace/AD/data/mvtec"
```

表示这次 datamodule 实际读取的数据根目录。

也就是说，后面的所有 `bottle/train/good/...`、`bottle/test/...` 都是从这个目录下面找出来的。

### `category`

```json
"category": "bottle"
```

表示当前示例正在读取哪个 MVTec 类别。

如果你运行：

```bash
uv run python examples/data_usage_demo.py --category cable
```

这里就会变成 `"cable"`。

### `available_categories`

```json
"available_categories": [
  "bottle",
  "cable",
  "capsule",
  ...
]
```

表示当前 `root` 目录下面实际存在的类别目录。

它只是一个“可用类别列表”，方便你知道这个 data root 里有哪些 MVTec 类别可以试。

### `split_lengths`

```json
"split_lengths": {
  "train": 127,
  "val": 41,
  "calibration": 41,
  "test": 83
}
```

表示 datamodule 最终构建出来的 4 个 split 各有多少样本。

这里的来源是：

- `train`：`train/good` 中保留下来用于真正训练的样本
- `val`：从 `train/good` 切出来的 held-out 验证子集
- `calibration`：从 `train/good` 切出来的 held-out 校准子集
- `test`：原始 `test/*` 下的样本

对于你这次运行：

- `val_split = 0.2`
- `calibration_split = 0.2`

所以 `train/good` 会先被拆成 3 块：

- 60% 留给训练
- 20% 给 `val`
- 20% 给 `calibration`

---

## `basic_demo`

这一段展示“最普通的数据使用方式”，也就是：

- 实例化 `MVTecDataModule`
- 调用 4 个 dataloader
- 各取一个 sample 看结构

### `basic_demo.config`

```json
"config": {
  "reference_index": 1,
  "val_split": 0.2,
  "calibration_split": 0.2,
  "split_seed": 7
}
```

这几项是这次演示最关键的 data 层配置。

#### `reference_index`

表示：

> 在“保留下来的训练子集”里，选第几个样本当 fixed reference。

注意这里不是从完整 `train/good` 里选，而是从 **retained train subset** 里选。

这是这次我们刚修过的一个重要语义。

#### `val_split`

表示从 `train/good` 拿出多少比例给验证集。

#### `calibration_split`

表示从 `train/good` 拿出多少比例给校准集。

#### `split_seed`

表示切分 `train / val / calibration` 时用的随机种子。

它只控制“哪些样本属于哪个 split”，不控制 dataloader 每次迭代时的运行时 shuffle 行为。

---

## `basic_demo.samples`

这一段会给出四个 split 各自取到的一个样本快照：

- `train`
- `val`
- `calibration`
- `test`

每个 sample 里的字段含义是一样的。

下面先解释这些通用字段，再解释你这次输出里几个容易误解的点。

### `sample_id`

例如：

```json
"sample_id": "train/good/162.png"
```

表示样本在当前 category 根目录下的相对路径。

它是一个稳定的样本标识，比纯文件名更有信息量。

因为 `000.png` 这种名字，在 `train` 和 `test` 下都可能同时存在。

### `label`

例如：

```json
"label": 0
```

或：

```json
"label": 1
```

在当前实现里：

- `0` 表示正常样本
- `1` 表示异常样本

所以你这次输出里：

- `train / val / calibration` 都来自 `train/good`，所以是 `0`
- `test` 示例取到了 `test/broken_large/000.png`，所以是 `1`

### `split`

例如：

```json
"split": "train"
```

或：

```json
"split": "val"
```

表示这条 sample 在 data contract 里属于哪个 split。

这是这次 data 层修复后的关键语义之一：

- `train` 一定是 `"train"`
- `val` 一定是 `"val"`
- `calibration` 一定是 `"calibration"`
- `test` 一定是 `"test"`

### `image_name`

例如：

```json
"image_name": "162.png"
```

表示当前样本图片文件的文件名。

它只是 `image_path` 的末尾文件名，方便看输出，不是全路径。

### `reference_name`

例如：

```json
"reference_name": "005.png"
```

表示当前样本携带的 fixed reference 图片文件名。

这个字段很重要，因为它告诉你：

> 所有 split 当前都在和哪一张参考正常图比较。

你这次输出里，4 个 split 的 `reference_name` 都是 `005.png`，说明：

- train/val/calibration/test 共用同一个 fixed reference
- 而且这个 reference 已经被约束在 retained train subset 里，不会来自 held-out 样本

### `image_type`

例如：

```json
"image_type": "Tensor"
```

表示当前 `sample.image` 的 Python 类型。

这里是 `Tensor`，因为示例里用了默认 sample transform，把原始 PIL 图片转成了 tensor。

### `image_shape`

例如：

```json
"image_shape": [3, 32, 32]
```

表示当前 `sample.image` 的张量形状。

含义是：

- `3` 个通道
- 高度 `32`
- 宽度 `32`

之所以是 `32x32`，是因为这个示例脚本里显式用了：

```python
image_size=(32, 32)
```

### `reference_type`

例如：

```json
"reference_type": "Tensor"
```

表示 `sample.reference` 的类型。

这里和 `image_type` 一样也是 `Tensor`，说明 reference 也经过了同样的 transform。

### `reference_shape`

例如：

```json
"reference_shape": [3, 32, 32]
```

表示 `sample.reference` 的张量形状。

它通常和 `image_shape` 一致，因为当前 transform 会对 image 和 reference 应用同样的 resize / tensorize 策略。

### `has_reference`

例如：

```json
"has_reference": true
```

表示这条 sample 是否携带了 reference。

你这次输出里全是 `true`，因为当前 MVTec datamodule 会为所有 sample 提供 fixed reference。

---

## 为什么 `train` 样本不是 `000.png`？

这是最容易让人困惑的地方。

你这次看到：

```json
"train": {
  "sample_id": "train/good/162.png",
  ...
}
```

这不表示训练集只从 `162.png` 开始，也不表示排序错了。

原因是：

- `train_dataloader()` 默认是 `shuffle=True`
- 所以你取 `next(iter(train_dataloader()))` 时，拿到的是“当前迭代顺序中的第一条”，不是“按文件名排序的第一条”

而：

- `val_dataloader()` / `calibration_dataloader()` / `test_dataloader()` 默认不 shuffle

所以它们看起来更像稳定顺序。

一句话说：

> `train` 示例展示的是“训练时拿到的一条样本”，不是“字典序第一张图片”。

---

## 为什么 `reference_name` 是 `005.png`？

你这次配置是：

```json
"reference_index": 1
```

它的意思不是：

> 直接取 `train/good` 全量排序后的第 2 张

而是：

> 先按 `split_seed=7` 把 `train/good` 切成 `train / val / calibration`
> 再在保留下来的 train 子集里，按有序顺序取第 2 张

所以最后 reference 落到 `005.png` 是正常的。

这里体现的是当前 data 层修过后的两个规则：

1. fixed reference 不能来自 held-out split
2. no-holdout 情况下，`reference_index` 仍保持稳定有序语义

---

## `transform_demo`

这一段专门不是为了展示 tensor，而是为了展示：

> 自定义 transform 执行时，到底看到了什么 `split`

这是这次 data 层修复的另一个关键点。

### `transform_demo.observed_splits`

例如：

```json
"observed_splits": {
  "val": ["val"],
  "calibration": ["calibration"]
}
```

这表示示例里那个自定义 transform 实际运行时，记录到了哪些 split。

这里说明：

- 当 val sample 进入 transform 时，它看到的是 `"val"`
- 当 calibration sample 进入 transform 时，它看到的是 `"calibration"`

而不是旧行为里的 `"train"`

### `transform_demo.samples.val.observed_split_before_transform`

例如：

```json
"observed_split_before_transform": "val"
```

这个字段是示例 transform 人工塞回 sample metadata 里的。

它的目的是证明：

> transform 执行之前，这个 sample 的 split 已经被 relabel 成正确值了

同理：

```json
"observed_split_before_transform": "calibration"
```

表示 calibration 样本在 transform 看到的也是正确 split。

---

## 这份输出实际证明了什么

如果把你这次 JSON 压缩成几个结论，就是：

1. data 层现在能稳定产出 4 个 split：
   - `train`
   - `val`
   - `calibration`
   - `test`

2. `val/calibration` 不再伪装成 `train`

3. 所有 split 共用同一个 fixed reference

4. 这个 fixed reference 来自 retained train subset，而不是 held-out 样本

5. 自定义 transform 在 `val/calibration` 上看到的 split 已经是正确语义

---

## 最后补一个小提醒

你终端里这行：

```text
进程已结束，退出代码为 0
```

不是 JSON 的一部分，它只是 IDE/终端告诉你：

> 脚本正常执行结束，没有报错

也就是说，这次 `examples/data_usage_demo.py` 是成功跑通的。
