import torch
from torch_scatter import scatter_max

src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
index = torch.tensor([[4, 5, 4, 2, 3], [0, 0, 2, 2, 1]])
out = src.new_zeros((2, 5))

out, argmax = scatter_max(src, index, out=out)

print(out)

{"documents":
[{"is_selected": true,
  "title": "在网上下载字体后怎么打开安装",
  "most_related_para": 2,
  "segmented_title": ["在", "网上", "下载", "字体", "后", "怎么", "打开", "安装"],
  "segmented_paragraphs": [["<", "p", ">", "图片", "展示", "的", "字体", "文件", "的", "样子", "<", "/", "p", ">", "<", "p", ">", "解药", "你", "的", "压缩包", "后", "，", "你", "可以看到", "类似", "的", "文件", "<", "/", "p", ">", "<", "p", ">", "把", "这些", "文件", "复制", "到", "&", "nbsp", ";", "&", "nbsp", ";", "控制面板", "-", "-", "-", "》", "字体", "&", "nbsp", ";", "&", "nbsp", ";", "&", "nbsp", ";", "&", "nbsp", ";", "这个", "文件夹", "里面", "即可", "<", "/", "p", ">", "<", "p", ">", "<", "img", "src", "=", "\"", "1626805363", "\"", "/", ">", "<", "/", "p", ">"],
                           ["一般", "下载", "的", "字体", "文件", "都是", "压缩包", "，", "将", "你", "下载", "的", "压缩包", "解压缩", "，", "会", "得到", "一", "个", "后缀", "为", ".", "ttf", "的", "字体", "文件", "，", "将", "这个", "文件", "直接", "复制", "到", "字体", "文件夹", "（", "Fonts", "）", "里", "就是", "了", "。", "字体", "文件夹", "的", "位置", "为", "：", "C", ":", "\\", "WINDOWS", "\\", "Fonts", "，", "或者", "在", "控制面板", "里", "也有", "一", "个", "字体", "文件夹", "。", "（", "千万不要说", "你", "不会", "解压缩", "啊", "）"],
                           ["<", "p", ">", "方法", "一", "：", "<", "br", "/", ">", "<", "/", "p", ">", "<", "ol", ">", "<", "li", ">", "<", "p", ">", "直接", "双击", "字体", "文件", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "点击", "并", "等待", "安装", "完成", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "/", "ol", ">", "<", "p", ">", "方法", "二", "：", "<", "/", "p", ">", "<", "ol", ">", "<", "li", ">", "<", "p", ">", "打开", "文件夹", "路径", "C", ":", "\\", "Windows", "\\", "Fonts", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "将", "字体", "文件", "复制", "或", "粘贴", "到", "此", "目录", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "/", "ol", ">", "<", "p", ">", "方法", "三", "：", "<", "/", "p", ">", "<", "ol", ">", "<", "li", ">", "<", "p", ">", "打开", "控制面板", "。", "（", "win7", "以下", "系统", "可", "直接", "在", "开始菜单", "打开", "，", "win7", "以上", "可", "按", "Win", "+", "X", "组合键", "，", "再点击", "控制面板", "）", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "找到", "字体", "选项", "并", "点击", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "将", "字体", "文件", "复制", "或", "粘贴", "到", "此", "目录", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "/", "ol", ">", "<", "p", ">", "注意", "：", "<", "/", "p", ">", "<", "ol", ">", "<", "li", ">", "<", "p", ">", "字体", "文件", "后缀", "大", "多", "为", "ttf", "，", "不要", "认错", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "下载", "字体", "多", "为", "压缩包", "，", "需", "解压", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "不要", "随意", "更改", "字体", "文件夹", "的", "字体", "文件", "，", "否则", "可能", "会", "产生", "系统", "错误", "<", "br", "/", ">", "<", "/", "p", ">", "<", "/", "li", ">", "<", "/", "ol", ">"], ["把", "解压缩", "之后", "的", "文件", ",", "放到", "C", ":", "\\", "WINDOWS", "\\", "Fonts", "下", "确认", "你", "下", "的", "压缩文件", "没有问题", "吧", ",", "大小", "3", ".", "91", "M", ",", "解开", "以后", "不会", "没", "东西"]], "paragraphs": ["<p>图片展示的字体文件的样子</p><p>解药你的压缩包后，你可以看到类似的文件</p><p>把这些文件复制到&nbsp;&nbsp;控制面板---》字体&nbsp;&nbsp;&nbsp;&nbsp;这个文件夹里面即可</p><p><img src=\"1626805363\" /></p>", "一般下载的字体文件都是压缩包，将你下载的压缩包解压缩，会得到一个后缀为.ttf的字体文件，将这个文件直接复制到字体文件夹（Fonts）里就是了。字体文件夹的位置为：C:\\WINDOWS\\Fonts，或者在控制面板里也有一个字体文件夹。（千万不要说你不会解压缩啊）", "<p>方法一：<br /></p><ol><li><p>直接双击字体文件。</p></li><li><p>点击并等待安装完成。</p></li></ol><p>方法二：</p><ol><li><p>打开文件夹路径C:\\Windows\\Fonts。</p></li><li><p>将字体文件复制或粘贴到此目录。</p></li></ol><p>方法三：</p><ol><li><p>打开控制面板。（win7以下系统可直接在开始菜单打开，win7以上可按Win+X组合键，再点击控制面板）</p></li><li><p>找到字体选项并点击。</p></li><li><p>将字体文件复制或粘贴到此目录。</p></li></ol><p>注意：</p><ol><li><p>字体文件后缀大多为ttf，不要认错。</p></li><li><p>下载字体多为压缩包，需解压。</p></li><li><p>不要随意更改字体文件夹的字体文件，否则可能会产生系统错误<br /></p></li></ol>", "把解压缩之后的文件, 放到 C:\\WINDOWS\\Fonts下确认你下的压缩文件没有问题吧, 大小3.91M, 解开以后不会没东西"]}, {"is_selected": false, "title": "怎么安装自己下载的字体", "most_related_para": 1, "segmented_title": ["怎么", "安装", "自己", "下载", "的", "字体"], "segmented_paragraphs": [["直接", "将", "下载", "的", "字体", "文件", "，", "一般", "是", "TTF", "结尾", "的", "文件", "放到", "C盘", "WINDOWS", "文件夹", "下", "的", "fonts", "文件夹", "里", "就", "可以", "了", ".", "找不到", "？", "你", "是", "在", "什么", "软件", "里", "使用", "啊", "？"], ["<", "ol", ">", "<", "li", ">", "<", "p", ">", "直接", "把", "字体", "文件", "复制", "到", "系统", "字体", "文件夹", "C", ":", "\\", "Windows", "\\", "Fonts", "里面", "就", "安装", "成功", "了", "，", "注意", "是", "字体", "文件", "，", "不要", "是", "压缩包", "。", "<", "/", "p", ">", "<", "p", ">", "<", "img", "src", "=", "\"", "25051784610", "\"", "/", ">", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "在", "win7", "或", "win8", "系统", "下", "，", "可以通过", "右键", "点击", "字体", "文件", "，", "选择", "安装", "，", "即可", "安装", "字体", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "/", "ol", ">"], ["将", "下载", "的", "字体", "文件", "*", ".", "TTF", "或者", "*", ".", "FON", "文件", "复制", "到", "C", ":", "\\", "WINDOWS", "\\", "Fonts", "目录", "下面", "会", "自动", "安装", "的", "。"]], "paragraphs": ["直接将下载的字体文件，一般是TTF结尾的文件放到C盘WINDOWS文件夹下的fonts文件夹里就可以了 .找不到？你是在什么软件里使用啊？", "<ol><li><p>直接把字体文件复制到系统字体文件夹C:\\Windows\\Fonts里面就安装成功了，注意是字体文件，不要是压缩包。</p><p><img src=\"25051784610\" /></p></li><li><p>在win7或win8系统下，可以通过右键点击字体文件，选择安装，即可安装字体。</p></li></ol>", "将下载的字体文件*.TTF 或者*.FON 文件复制到C:\\WINDOWS\\Fonts目录下面会自动安装的。"]},

 {"is_selected": false,
  "title": "怎么在电脑上下载新字体,下载后怎么安装?",
  "most_related_para": 0,
  "segmented_title": ["怎么", "在", "电脑", "上", "下载", "新", "字体", ",", "下载", "后", "怎么", "安装", "?"],
  "segmented_paragraphs": [["。", "。", "。", "或者", "在", "下载", "守", "字体", "包", "或者", "文件", "后", "，", "打开", "字体", "文件夹", "(", "C", ":", "\\", "WINDOWS", "\\", "Fonts", ")", "-", "-", "-", "文件", "菜单", "-", "-", "-", "安装", "新", "字体", "-", "-", "-", "选择", "字体", "所在", "的", "文件夹", "-", "-", "-", "-", "-", "然后", "把", "将", "字体", "复制", "到", "fontS", "文件夹", "的", "勾", "去掉", ".", "这样", "就", "可以", "了", "。", "开始", "-", "控制面板", "-", "字体", "然后", "把", "你", "下载", "的", "字体", "文件", "拉", "进去", "，", "就", "可以", "了", "，", "在", "某些", "程序", "选择", "字体", "就", "可以看到", "你", "下载", "的", "字体", "了", "，"]],
  "paragraphs": ["。。。或者在下载守字体包或者文件后，打开字体文件夹(C:\\WINDOWS\\Fonts)---文件菜单---安装新字体---选择字体所在的文件夹-----然后把将字体复制到fontS文件夹的勾去掉.这样就可以了。开始-控制面板-字体 然后把你下载 的 字体文件 拉进去， 就可以了，  在某些程序选择字体就可以看到你下载的字体了，"]},
 {"is_selected": false,
  "title": "下载完的字体怎么安装啊?急!",
  "most_related_para": 1,
  "segmented_title": ["下载", "完", "的", "字体", "怎么", "安装", "啊", "?", "急", "!"],
  "segmented_paragraphs": [["最好", "是", "把", "一些", "安装", "进", "的", "辅助", "程序", "装", "在", "别", "的", "盘", "里", "。", "。", "即使", "是", "重装系统", "也", "不会", "还要", "设置", "，", "你可以", "在", "安装", "在", "别", "的", "盘", "里", "。", "。", "看仔细", "一", "点", "。", "。", "一步一步", "设置", "。", "。", "如果", "是", "字体", "。", "。", "看", "格式", "吧", "。", "。", "前面", "那些", "格式", "怎么", "做", "出来", "的", "，", "，", "你", "调", "成", "那样", "的", "格式", "试", "试"],
                           ["这些", "简单", "环节", "不应该", "出错", "呀", "，", "正常", "安", "完", "就", "好使", "了", "呀", "。", "1", ",", "检查", "你", "的", "字体", "文件", "是否", "是", "好", "的", "。", "2", ",", "安装", "字体", "是否", "正确", "：", "复制", "字体", "XXX", ".", "ttf", "到", "c", ":", "\\", "windows", "\\", "fonts", "下", "（", "这个", "你", "已经", "做", "了", "）", "，", "安装", "时", "有", "提示", "吗", "？", "是否", "成功", "？", "再试试", "这样", "安装", "：", "开始", "→", "控制面板", "→", "切换", "到", "经典", "视图", "→", "打开", "字体", "选项", "→", "文件", "→", "安装", "新", "字体", "→", "找到", "你", "的", "新", "字体", "的", "路径", "安装", "成功", "后", "，", "应该", "调用", "系统", "字体", "的", "软件", "都", "能", "使用", "了", "。", "3", ",", "word", "问题", "，", "（", "可能", "很", "小", "）", "祝", "好运", "！"],
                           ["<", "p", ">", "win7", "系统", "安装", "字体", "安装步骤", "：", "<", "/", "p", ">", "<", "ol", ">", "<", "li", ">", "<", "p", ">", "首先", "下载", "对应", "字体", "安装包", "，", "一般", "为", "压缩包", "，", "因此", "需", "先", "解压", "字体", "压缩包", "到", "指定", "文件夹", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "右键", "点击", "字体", "文件", "，", "选择", "“", "安装", "”", "即可", "。", "<", "/", "p", ">", "<", "/", "li", ">", "<", "li", ">", "<", "p", ">", "等候", "安装", "完成", "即可", "。", "<", "/", "p", ">", "<", "p", ">", "<", "/", "p", ">", "<", "/", "li", ">", "<", "/", "ol", ">"]],
  "paragraphs": ["最好是把一些安装进的辅助程序装在别的盘里。。即使是重装系统也不会还要设置，你可以在安装在别的盘里。。看仔细一点。。一步一步设置。。 如果是字体。。看格式吧。。前面那些格式怎么做出来的，，你调成那样的格式试试",
                 "这些简单环节不应该出错呀，正常安完就好使了呀。1,检查你的字体文件是否是好的。2,安装字体是否正确：复制字体XXX.ttf到c:\\windows\\fonts下（这个你已经做了），安装时有提示吗？是否成功？再试试这样安装：开始→控制面板→切换到经典视图→打开字体选项→文件→安装新字体→找到你的新字体的路径安装成功后，应该调用系统字体的软件都能使用了。3,word问题，（可能很小）祝好运！", "<p>win7系统安装字体安装步骤：</p><ol><li><p>首先下载对应字体安装包，一般为压缩包，因此需先解压字体压缩包到指定文件夹。</p></li><li><p>右键点击字体文件，选择“安装”即可。</p></li><li><p>等候安装完成即可。</p><p></p></li></ol>"]}, {"is_selected": false, "title": "下载了字体后怎么安装到ppt?", "most_related_para": 3, "segmented_title": ["下载", "了", "字体", "后", "怎么", "安装", "到", "ppt", "?"], "segmented_paragraphs": [["字体", "也是", "要", "安装", "的", "，", "，", "怎么", "复制", "？", "开始", "-", "-", "-", "设置", "-", "-", "-", "控制面板", "-", "-", "-", "-", "-", "-", "字体", "。", "文件", "-", "-", "-", "安装", "新", "字体", "。", "。", "安装", "好", "了", "才能", "在", "PPT", "里面", "调用", "，", "，", "，"],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ["字体", "安装", "的", "话", "直接", "双击", "进行", "安装", "，", "再", "重启", "powerpoint"], ["按照", "一", "楼", "的", "方法", "，", "先", "把", "字体", "装", "在", "系统", "里", "，", "重启", "之后", "PPT", "里", "，", "会", "自动", "加载", "你", "新", "加入", "的", "字体", "，", "选择", "即可", "使用", "！"], ["先", "把", "PPT", "关", "了", "。", "。", "不", "然", "装", "不好", "的", "。", "。", "然后", "把", "下", "的", "字体", "。", "。", "点击", "开始", "-", "-", "-", "设置", "-", "-", "-", "控制面板", "-", "-", "-", "-", "-", "-", "字体", "。", "文件", "-", "-", "-", "安装", "新", "字体", "。", "。", "在", "字体", "选项", "那", "一栏", "，", "找到", "你", "下载", "的", "字体", "、", "、", "这样", "之后", "就", "可以", "用", "了", "。", "。", "望", "采纳", "。"]], "paragraphs": ["字体也是要安装的，，怎么复制？开始---设置---控制面板------字体。文件---安装新字体。。安装好了才能在PPT里面调用，，，", "字体安装的话直接双击进行安装，再重启powerpoint", "按照一楼的方法，先把字体装在系统里，重启之后PPT里，会自动加载你新加入的字体，选择即可使用！", "先把PPT关了。。不然装不好的。。然后把下的字体。。点击开始---设置---控制面板------字体。文件---安装新字体。。在字体选项那一栏，找到你下载的字体、、这样之后就可以用了。。望采纳。"]}], "answer_spans": [[79, 212]], "fake_answers": ["打开文件夹路径C:\\Windows\\Fonts。</p></li><li><p>将字体文件复制或粘贴到此目录。</p></li></ol><p>方法三：</p><ol><li><p>打开控制面板。（win7以下系统可直接在开始菜单打开，win7以上可按Win+X组合键，再点击控制面板）</p></li><li><p>找到字体选项并点击。</p></li><li><p>将字体文件复制或粘贴到此目录。"], "question": "下载字体如何安装", "segmented_answers": [["方法", "一", "：", "直接", "双击", "字体", "文件", "，", "点击", "并", "等待", "安装", "完成", "。", "方法", "二", "：", "打开", "文件夹", "路径", "C", ":", "\\", "Windows", "\\", "Fonts", "，", "将", "字体", "文件", "复制", "或", "粘贴", "到", "此", "目录", "。", "方法", "三", "：", "打开", "控制面板", "。", "（", "win7", "以下", "系统", "可", "直接", "在", "开始菜单", "打开", "，", "win7", "以上", "可", "按", "Win", "+", "X", "组合键", "，", "再点击", "控制面板", "）", "，", "找到", "字体", "选项", "并", "点击", "。", "将", "字体", "文件", "复制", "或", "粘贴", "到", "此", "目录", "。"]], "answers": ["方法一：直接双击字体文件，点击并等待安装完成。方法二：打开文件夹路径C:\\Windows\\Fonts，将字体文件复制或粘贴到此目录。方法三：打开控制面板。（win7以下系统可直接在开始菜单打开，win7以上可按Win+X组合键，再点击控制面板），找到字体选项并点击。将字体文件复制或粘贴到此目录。"], "answer_docs": [0], "segmented_question": ["下载", "字体", "如何", "安装"], "question_type": "DESCRIPTION", "question_id": 186974, "fact_or_opinion": "FACT", "match_scores": [0.5925925925925926]}
