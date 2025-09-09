library(data.table)
library(ggplot2)
library(patchwork)
library(scales)
files = list.files("./build/res", full.names = TRUE)

res_dt = rbindlist(lapply(files, \(x) fread(x)))
setnames(res_dt, "# method", "method")
res_dt[, bench := NA_character_]
res_dt[grepl("var_eigen", name), bench := "SoA"]
res_dt[grepl("lambda_eigen_special", name), bench := "AoS Overload"]
res_dt[grepl("lambda_eigen_bench", name), bench := "AoS"]
res_dt[ , n_dim := as.numeric(strsplit(name, "/")[[1]][2]), .I]

base_dt = res_dt[bench == "SoA"]
res_dt[, scaled_time := 1]
res_dt[bench == "AoS Overload", scaled_time := cpu_time / base_dt[, cpu_time]]
res_dt[bench == "AoS", scaled_time := cpu_time / base_dt[1:10, cpu_time]]
res_dt[, bench := as.factor(bench)]
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
def_colors = gg_color_hue(3)
group_colors = c(
  "AoS Overload" = def_colors[1],
  "AoS" = def_colors[2],
  "SoA" = def_colors[3]
)
p1 = ggplot(res_dt[n_dim > 4], aes(x = n_dim, y = scaled_time, color = bench)) +
  geom_line() +
  geom_point() +
  scale_y_log10(
    labels = label_number(scale_cut = cut_short_scale(), suffix = " x"), n.breaks = 5
  ) +
  scale_x_log10(
    labels = label_number(scale_cut = cut_short_scale())
  ) +
  xlab("N") +
  ylab("") +
  scale_color_manual(values = group_colors) +
  theme_bw()
p1
p2 = ggplot(res_dt[n_dim > 4 & bench != "AoS"],
  aes(x = n_dim, y = scaled_time, color = bench)) +
  geom_line() +
  geom_point() +
  scale_y_log10(
    labels = label_number(scale_cut = cut_short_scale(), suffix = " x"), n.breaks = 5
  ) +
  scale_x_log10(
    labels = label_number(scale_cut = cut_short_scale())
  ) +
  xlab("N") +
  ylab("") +
  scale_color_manual(values = group_colors) +
  guides(color="none") +
  theme_bw()
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
ggsave(plot = combined_plot, filename =
    file.path(graph_path, "matmul_bench.png"),
  width = 9, height = 5.5, dpi = 300, units = "in")

