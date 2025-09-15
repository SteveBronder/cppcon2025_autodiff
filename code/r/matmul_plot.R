library(data.table)
library(ggplot2)
library(patchwork)
library(scales)
files = list.files("./build/res", full.names = TRUE)

res_dt = rbindlist(lapply(files, \(x) fread(x)))
res_dt[, bench := NA_character_]
res_dt[grepl("var_eigen", name), bench := "SoA"]
res_dt[grepl("lambda_eigen_special", name), bench := "AoS Overload"]
res_dt[grepl("lambda_eigen_bench", name), bench := "AoS"]
res_dt[grepl("expr_template", name), bench := "SCT"]
res_dt = res_dt[bench != "AoS Overload"]
res_dt[ , n_dim := as.numeric(strsplit(name, "/")[[1]][2]), .I]
res_dt[, .SD[which.max(n_dim)], bench]
base_dt = res_dt[bench == "SCT"]
res_dt[, scaled_time := 1]
res_dt[bench != "AoS", scaled_time := cpu_time / base_dt[, cpu_time]]
res_dt[bench == "AoS", scaled_time := cpu_time / base_dt[1:10, cpu_time]]
summary(res_dt)
res_dt[, bench := as.factor(bench)]
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
def_colors = gg_color_hue(4)
group_colors = c(
  "AoS Overload" = def_colors[1],
  "AoS" = def_colors[2],
  "SoA" = def_colors[3],
  "SCT" = def_colors[4]
)
p1 = ggplot(res_dt[n_dim > 4], aes(x = n_dim, y = scaled_time, color = bench)) +
  geom_line() +
  geom_point() +
  scale_y_log10(
    labels = label_number(scale_cut = cut_short_scale(), suffix = " x"), n.breaks = 8
  ) +
  scale_x_log10(
    labels = label_number(scale_cut = cut_short_scale()), n.breaks = 8,
  ) +
  xlab("") +
  ylab("") +
  scale_color_manual(values = group_colors) +
  theme_bw() +
  theme(legend.position = "bottom", legend.title=element_blank(),
    text = element_text(size = 20))

p1
p2 = ggplot(res_dt[n_dim > 4 & bench != "AoS"],
  aes(x = n_dim, y = scaled_time, color = bench)) +
  geom_line() +
  geom_point() +
  scale_y_log10(
    labels = label_number(scale_cut = cut_short_scale(), suffix = " x"), n.breaks = 8
  ) +
  scale_x_log10(
    labels = label_number(scale_cut = cut_short_scale()), n.breaks = 8,
  ) +
  xlab("") +
  ylab("") +
  scale_color_manual(values = group_colors) +
#  guides(color="none") +
  theme_bw() +
  theme(legend.position = "bottom", legend.title=element_blank(),
    text = element_text(size = 20))

p2
combined_plot = p1 +
  p2 &
  theme(legend.position = "bottom", legend.title=element_blank(),
    text = element_text(size = 16))
combined_plot = combined_plot +
#  plot_annotation(title = "Relative Performance Matrix Multiplication Benchmark") +
  plot_layout(guides = "collect")
combined_plot

graph_path = "./slides/img"
ggsave(plot = p1, filename =
    file.path(graph_path, "matmul_bench_slow.png"),
  width = 9, height = 5.5, dpi = 300, units = "in")


ggsave(plot = p2, filename =
    file.path(graph_path, "matmul_bench_fast.png"),
  width = 9, height = 5.5, dpi = 300, units = "in")
