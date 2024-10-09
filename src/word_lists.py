# Python list of Vietnamese nouns referring to individuals and groups of people (extended)
PEOPLE = [
    # Danh từ chỉ người cá nhân
    "người", "người_đàn_ông", "người_phụ_nữ", "em_bé", "trẻ_con", "thiếu_niên", 
    "thanh_niên", "người_già", "cụ_ông", "cụ_bà", "anh", "chị", "ông", "bà", 
    "cậu", "mợ", "chú", "bác", "dì", "em", "thầy", "cô", "bạn", "người_yêu", 
    "đồng_nghiệp", "giáo_sư", "tiến_sĩ", "nhà_khoa_học", "bác_sĩ", "luật_sư", 
    "kỹ_sư", "công_nhân", "học_sinh", "sinh_viên", "tài_xế", "ca_sĩ", "diễn_viên", 
    "người_mẫu", "vận_động_viên", "nhà_văn", "nhà_thơ", "nhà_báo", "phóng_viên", 
    "họa_sĩ", "nhạc_sĩ", "nhà_thiết_kế", "doanh_nhân", "nông_dân", "ngư_dân", 
    "thợ_sửa_xe", "thợ_điện", "thợ_xây", "người_hùng", "tội_phạm", "người_nổi_tiếng", 
    "chính_trị_gia", "lãnh_đạo", "người_quản_lý", "giám_đốc", "phụ_tá", "thư_ký", 
    "nhân_viên", "nhà_nghiên_cứu", "nhà_kinh_tế_học", "nhà_xã_hội_học", "kẻ_cướp", 
    "người_bảo_vệ", "vệ_sĩ", "người_giám_sát", "người_quan_sát", "người_hướng_dẫn", 
    "trưởng_phòng", "người_phát_ngôn", "người_quản_trị", "giáo_viên", "giám_thị", 
    "thợ_mộc", "thợ_rèn", "thợ_gốm", "thợ_máy", "thợ_hàn", "thợ_lặn", "phi_công", 
    "tiếp_viên_hàng_không", "nhà_nghiệp_dư", "người_đại_diện", "người_học_việc", 
    "cảnh_sát", "thám_tử", "nhân_viên_tư_vấn", "nhân_viên_y_tế", "nha_sĩ", 
    "y_tá", "kế_toán", "thủ_kho", "nhân_viên_hải_quan", "công_chứng_viên", 
    "đại_biểu_quốc_hội", "nghệ_sĩ_điện_ảnh", "nhà_đạo_diễn", "nhà_sản_xuất_phim", 
    "diễn_viên_hài", "nhà_ảo_thuật", "kẻ_xấu", "người_tị_nạn", "người_công_giáo", 
    "người_hồi_giáo", "nhà_tu_hành", "nhà_ngoại_giao", "người_tiêu_dùng", "người_nông_dân",
    "đàn_ông", "phụ_nữ"
    
    # Danh từ chỉ nhóm người
    "đội", "nhóm", "băng", "bộ_tộc", "gia_đình", "họ_hàng", "bạn_bè", 
    "đồng_nghiệp", "cộng_đồng", "đám_đông", "tập_thể", "tổ_chức", 
    "đoàn_thể", "công_ty", "ban_nhạc", "tập_đoàn", "nhóm_nhạc", "nhà_trường", 
    "phòng_ban", "đội_quân", "tiểu_đội", "trung_đội", "đại_đội", "binh_đoàn", 
    "quân_đội", "đảng", "chính_phủ", "quốc_hội", "hội_đồng", "ủy_ban", 
    "ban_chấp_hành", "đội_bóng", "hội_nhóm", "phái_đoàn", "ban_điều_hành", 
    "ban_lãnh_đạo", "liên_đoàn", "đội_đặc_nhiệm", "tổ_công_tác", "hội_đồng_quản_trị",

    # Danh từ chỉ mối quan hệ gia đình
    "bố", "mẹ", "cha", "má", "ba", "ông", "bà", "chú", "bác", "dì", "cậu", 
    "mợ", "thím", "chồng", "vợ", "con_trai", "con_gái", "anh_trai", "em_trai", 
    "chị_gái", "em_gái", "cháu_ngoại", "cháu_nội", "cháu_trai", "cháu_gái", 
    "bố_chồng", "mẹ_chồng", "bố_vợ", "mẹ_vợ", "anh_rể", "chị_dâu", "em_dâu", 
    "em_rể", "ông_nội", "bà_nội", "ông_ngoại", "bà_ngoại", "cháu_ngoại_trai", 
    "cháu_nội_trai", "dượng", "cô_dâu", "chú_rể",

    # Danh từ chỉ nhóm nghề nghiệp
    "nhân_viên_công_sở", "công_nhân_xây_dựng", "nhóm_công_nhân", "đội_kỹ_sư", 
    "nhóm_sinh_viên", "nhóm_nghiên_cứu", "ban_lãnh_đạo", "ban_quản_trị", 
    "tổ_chức_phi_chính_phủ", "nhóm_thiện_nguyện", "đoàn_đại_biểu", "ban_phát_triển",
    "nhóm_văn_nghệ", "ban_đào_tạo", "đội_kiểm_lâm", "đội_chữa_cháy"
]
PLACE = [
    # Địa điểm hành chính, tự nhiên, và cư trú
    "quốc_gia", "thành_phố", "thị_trấn", "xã", "phường", "quận", "tỉnh", "khu_phố",
    "tòa", "toà", "nhà", "biên_giới", "bãi_biển", "đảo", "núi", "khu_rừng", "cánh_đồng", 
    "đồng_bằng", "đầm_lầy", "sông", "suối", "hồ", "vịnh", "thác_nước", "hang_động", 
    "vườn_quốc_gia", "khu_bảo_tồn", "vùng_đồng_bằng_sông_cửu_long", 
    "vùng_đông_nam_bộ", "cao_nguyên", "thảo_nguyên", "hoang_mạc", "sa_mạc", 
    "vùng_lãnh_thổ", "vườn_hoa", "khu_dân_cư", "khu_nhà_ở", "khu_tái_định_cư", 
    "khu_chung_cư", "chung_cư", "nhà_trọ", "thôn", "ấp", "làng", "xóm", "bản", 
    "khu_trại", "khu_căn_hộ", "khu_vực_cấm",
]

ORGANIZATION = [
    # Địa điểm giáo dục
    "trường_học", "trường_mẫu_giáo", "trường_tiểu_học", "trường_trung_học_cơ_sở", 
    "trường_trung_học_phổ_thông", "trường_nghề", "đại_học", "cao_đẳng", "học_viện", 
    "trung_tâm_ngoại_ngữ", "trung_tâm_gia_sư", "trung_tâm_dạy_nghề", "trung_tâm_hướng_nghiệp",
    
    # Địa điểm chăm sóc sức khỏe
    "bệnh_viện", "phòng_khám", "trung_tâm_y_tế", "trạm_xá", "nhà_thuốc", "phòng_mạch", 
    "bệnh_viện_nhi", "bệnh_viện_phụ_sản", "bệnh_viện_đa_khoa", "bệnh_viện_chuyên_khoa",
    
    # Địa điểm hành chính, công quyền
    "ủy_ban_nhân_dân", "tòa_án", "sở_cảnh_sát", "công_an", "trại_giam", "hải_quan", 
    "đại_sứ_quán", "lãnh_sự_quán", "cửa_khẩu", "tổng_cục_thống_kê", "bộ_tài_nguyên", 
    "bộ_giáo_dục", "bộ_y_tế", "bộ_nội_vụ", "bộ_ngoại_giao",
    
    # Địa điểm kinh doanh, dịch vụ
    "siêu_thị", "chợ", "trung_tâm_thương_mại", "cửa_hàng", "cửa_hàng_bách_hóa", 
    "cửa_hàng_tiện_lợi", "nhà_hàng", "quán_cà_phê", "tiệm_bánh", "tiệm_làm_tóc", 
    "khách_sạn", "nhà_nghỉ", "resort", "văn_phòng", "công_ty", "tập_đoàn", 
    "công_ty_xuất_nhập_khẩu", "trụ_sở", "chi_nhánh", "kho_hàng", "kho_lưu_trữ", 
    "nhà_máy", "xưởng_sản_xuất", "khu_công_nghiệp", "khu_chế_xuất", "ngân_hàng", 
    "phòng_giao_dịch", "kho_bạc", "bưu_điện", "nhà_máy_điện", "trung_tâm_điện_máy", 
    "trạm_xăng", "nhà_đấu_giá", "công_ty_du_lịch", "hãng_hàng_không",
    
    # Địa điểm giải trí, văn hóa
    "quảng_trường", "công_viên", "sân_bóng", "sân_vận_động", "rạp_chiếu_phim", "rạp_hát", 
    "nhà_hát", "bảo_tàng", "thư_viện", "nhà_sách", "sân_khấu", "khu_vui_chơi", "sở_thú", 
    "thủy_cung", "nhà_trẻ", "trường_múa", "phòng_tranh", "vườn_thú", "trung_tâm_thể_thao",
    
    # Địa điểm tôn giáo, tín ngưỡng
    "nhà_thờ", "chùa", "đền", "miếu", "thánh_đường", "tháp", "tu_viện", 
    "chủng_viện", "đại_điện", "tịnh_xá", "trại_tị_nạn",
    
    # Địa điểm giao thông
    "sân_bay", "nhà_ga", "bến_xe", "bến_tàu", "trạm_xe_bus", "cảng", "bến_cảng", 
    "ga_tàu_điện", "trạm_thu_phí", "trạm_dừng_xe", "bãi_đỗ_xe", "bến_phà", 
    "khu_vực_trả_hàng", "trạm_giao_hàng", "bãi_đỗ_tàu"
]
number_map = {
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10
}