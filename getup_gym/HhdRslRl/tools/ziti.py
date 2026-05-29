import matplotlib.font_manager as fm
# 获取所有字体名，并过滤出包含‘Noto’的字体
all_fonts = [f.name for f in fm.fontManager.ttflist]
noto_fonts = [f for f in all_fonts if 'Noto' in f]
print("Matplotlib找到的Noto字体有：")
for font in sorted(set(noto_fonts))[:10]: # 打印前10个，避免刷屏
    print(f"  - {font}")
print(f"\n总计找到 {len(noto_fonts)} 个Noto字体变体。")