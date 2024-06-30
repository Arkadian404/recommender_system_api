from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate, ChatPromptTemplate, \
    SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings


class SQLAgent:
    model = "gpt-4o"
    examples = [
        {
            "input": "Liệt kê hết tất cả sản phẩm",
            "query": "SELECT * FROM product;"
        },
        {
            "input": "Liệt kê hết tất cả loại sản phẩm",
            "query": "SELECT * FROM category;"
        },
        {
            "input": "Có bao nhiêu loại sản phẩm trong cửa hàng",
            "query": "SELECT COUNT(id) as total FROM category;"
        },
        {
            "input": "Có bao nhiêu loại sản phẩm đang bán trong cửa hàng",
            "query": "SELECT COUNT(DISTINCT category_id) as total_categories FROM product;"
        },
        {
            "input": "Tìm tất cả sản phẩm thuộc loại cà phê đóng chai",
            "query": "SELECT p.name, c.name FROM product p join category c on p.category_id = c.id where c.name = 'cà phê đóng chai';",
        },
        {
            "input": "Cho biết các sản phẩm có hương vị trái cây",
            "query": "SELECT p.name, f.name FROM product p join flavor f on p.flavor_id = f.id where f.name = 'trái cây';",
        },
        {
            "input": "Tổng số lượng sản phẩm bán được",
            "query": "SELECT SUM(sold) as total_sold FROM product;",
        },
        {
            "input": "Liệt kê các khách hàng có địa chỉ ở Hà Nội",
            "query": "SELECT * FROM user WHERE province = 'Hà Nội';",
        },
        {
            "input": "Khách hàng có ID là 44 có tổng cộng bao nhiêu đơn hàng đã đặt?",
            "query": "SELECT COUNT(id) FROM `order` WHERE user_id = 44;",
        },
        {
            "input": "Liệt kê các thương hiệu cà phê nào bán được nhiều sản phẩm nhất",
            "query": "SELECT b.name, SUM(sold) as total_sold FROM product p JOIN brand b ON p.brand_id = b.id GROUP BY b.id;",
        },
        {
            "input": "Cho biết các sản phẩm cà phê có nguồn gốc từ Châu Mỹ",
            "query": "SELECT p.name FROM product p JOIN product_origin po ON p.product_origin_id = po.id WHERE po.continent LIKE '%Châu Mỹ%';",
        },
        {
            "input": "Top 5 khách hàng mua hàng nhiều nhất dựa trên tổng tiền của đơn hàng",
            "query": "SELECT user_id, full_name, SUM(total) AS total_purcharse FROM `order` GROUP BY user_id, full_name ORDER BY total_purcharse DESC LIMIT 5;",
        },
        {
            "input": "Liệt kê các sản phẩm ra mắt vào năm 2023",
            "query": "SELECT  p.name FROM product p WHERE  year(p.created_at) = 2023;",
        },
        {
            "input": "Có bao nhiêu đơn hàng trong hệ thống",
            "query": 'SELECT COUNT(id) as total FROM `order`;"',
        },
        {
            "input": "Tôi muốn mua cà phê rẻ của hệ thống, bạn hãy gợi ý cho tôi 1 vài sản phẩm",
            "query": 'SELECT  p.name, pd.weight, pd.price FROM product p JOIN product_detail pd ON p.id = pd.product_id ORDER BY price ASC LIMIT 5;"',
        },
        {
            "input": "Tôi muốn biết 5 sản phẩm được bán nhiều nhất trong cửa hàng",
            "query": "select p.name, p.sold from product p order by p.sold desc limit 5;"
        },
        {
            "input": "Liệt kê 5 sản phẩm rẻ nhất trong cửa hàng",
            "query": "select p.name, pd.weight, pd.price from product p join product_detail pd on p.id = pd.product_id order by pd.price asc limit 5;"
        },

        {
            "input": "Liệt kê tất cả các sản phẩm có giá từ 50.000 đến 100.000",
            "query": "SELECT distinct p.name, pd.price FROM product p JOIN product_detail pd ON p.id = pd.product_id WHERE pd.price BETWEEN 50000 AND 100000;"
        },
        {
            "input": "Tìm sản phẩm có đánh giá cao nhất",
            "query": "WITH ProductRatings AS ( SELECT p.id, p.name, ROUND(AVG(r.rating), 0) AS average_rating FROM product p JOIN review r ON p.id = r.product_id GROUP BY p.id, p.name ), MaxRating AS ( SELECT MAX(average_rating) AS max_rating FROM ProductRatings ) SELECT pr.id, pr.name, pr.average_rating FROM ProductRatings pr, MaxRating WHERE pr.average_rating = MaxRating.max_rating limit 10;"
        },
        {
            "input": "Số lượng sản phẩm từ mỗi loại sản phẩm",
            "query": "SELECT c.name, COUNT(p.id) FROM category c JOIN product p ON c.id = p.category_id GROUP BY c.name;"
        },
        {
            "input": "Liệt kê các sản phẩm không còn trong kho",
            "query": "SELECT p.name FROM product p JOIN product_detail pd ON p.id = pd.product_id WHERE pd.stock = 0;"
        },
        {
            "input": "Tìm tất cả sản phẩm có nguồn gốc từ Việt Nam",
            "query": "SELECT p.name FROM product p JOIN product_origin po ON p.product_origin_id = po.id WHERE po.name = 'Colombia';"
        },
        {
            "input": "Liệt kê các sản phẩm được ra mắt trong 3 tháng gần nhất",
            "query": "SELECT name FROM product WHERE created_at >= NOW() - INTERVAL 3 MONTH;"
        },
        {
            "input": "Sản phẩm nào có số lượng bán ra cao nhất trong 1 tháng trước",
            "query": "SELECT p.id as product_id, p.name as product_name, SUM(od.quantity) as total_quantity_sold FROM `order_detail` od JOIN product_detail pd ON od.product_detail_id = pd.id JOIN product p ON pd.product_id = p.id WHERE od.order_date >= DATE_FORMAT(CURRENT_DATE - INTERVAL 2 MONTH, '%Y-%m-01') AND od.order_date < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01') GROUP BY p.id, p.name ORDER BY  total_quantity_sold DESC LIMIT 1;"
        },
        {
            "input": "Sản phẩm nào có số lượng bán ra cao nhất trong 3 tháng trước",
            "query": "SELECT p.id as product_id, p.name as product_name, SUM(od.quantity) as total_quantity_sold FROM `order_detail` od JOIN product_detail pd ON od.product_detail_id = pd.id JOIN product p ON pd.product_id = p.id WHERE od.order_date >= DATE_FORMAT(CURRENT_DATE - INTERVAL 3 MONTH, '%Y-%m-01') AND od.order_date < DATE_FORMAT(CURRENT_DATE, '%Y-%m-01') GROUP BY p.id, p.name ORDER BY  total_quantity_sold DESC LIMIT 1;"
        },
        {
            "input": "Liệt kê tất cả các voucher còn hiệu lực",
            "query": "SELECT name, code, description FROM voucher WHERE expiration_date > NOW();"
        },
        {
            "input": "Các sản phẩm có hương vị trái cây",
            "query": "SELECT p.name FROM product p JOIN flavor f ON p.flavor_id = f.id WHERE f.name LIKE '%Trái cây%';"
        },
        {
            "input": "Cho biết số lượng tồn kho của sản phẩm Espresso",
            "query": "SELECT p.id as product_id, p.name as product_name, SUM(pd.stock) as total_stock FROM product p JOIN product_detail pd ON p.id = pd.product_id WHERE p.name LIKE '%Espresso%' GROUP BY p.id, p.name;"
        },
        {
            "input": "Hiện tại sản phẩm expresso có được giảm giá hay không?",
            "query": "(SELECT p.id, p.name, v.code FROM product p join voucher v on p.sale_id = v.id WHERE p.sale_id IS NOT NULL AND p.name LIKE '%espresso%') UNION (SELECT p.id, p.name, v.code FROM product p JOIN category c ON p.category_id = c.id JOIN voucher v ON c.id = v.category_id WHERE p.name LIKE '%Espresso%' AND v.code IS NOT NULL AND v.expiration_date > CURRENT_DATE);"
        },
        {
            "input": "Từ ngày 1 tháng 2 năm 2023, sản phẩm expresso có được giảm giá hay không?",
            "query": "((SELECT p.id, p.name, v.code FROM product p join voucher v on p.sale_id = v.id WHERE p.sale_id IS NOT NULL AND p.name LIKE '%espresso%') UNION (SELECT p.id, p.name, v.code FROM product p JOIN category c ON p.category_id = c.id JOIN voucher v ON c.id = v.category_id WHERE p.name LIKE '%Espresso%' AND v.code IS NOT NULL AND v.expiration_date > '2023-02-01');"
        },
        {
            "input": "Số lượng sản phẩm mỗi thương hiệu có?",
            "query": "SELECT b.name, COUNT(p.id) FROM brand b JOIN product p ON b.id = p.brand_id GROUP BY b.name;"
        },
        {
            "input": "Tìm sản phẩm có giá cao nhất?",
            "query": "SELECT p.name, pd.price, pd.weight, pd.stock FROM product_detail pd join product p on p.id = pd.product_id  ORDER BY price DESC LIMIT 1;"
        },
        {
            "input": "sản phẩm expresso có nguồn gốc từ đâu?",
            "query": "SELECT p.name as product, b.name as brand, po.name as origin FROM product p join brand b on p.brand_id = b.id join product_origin po on p.product_origin_id = po.id WHERE p.name LIKE '%espresso%';"
        },
        {
            "input": "Liệt kê tất cả các sản phẩm được đánh giá 5 sao",
            "query": "SELECT p.name, p.id FROM product p JOIN review r ON p.id = r.product_id where r.rating = 5 group by p.id;"
        },
        {
            "input": "Những sản phẩm Volcano nào có xuất xứ từ colombia?",
            "query": "select p.name from product p left join product_origin po on p.product_origin_id = po.id where p.name like '%volcano%' and po.name like '%Colombia%';"
        },
        {
            "input": "Sản phẩm Espresso đã bán được bao nhiêu và tồn kho bao nhiêu?",
            "query": "select p.name, p.sold, sum(pd.stock) as ton_kho from product_detail pd left join product p on pd.product_id = p.id where p.name like '%Espresso%' group by pd.product_id;"
        },
        {
            "input": "Số lượng sản phẩm từ mỗi nguồn gốc",
            "query": "SELECT po.name, COUNT(p.id) as so_luong_san_pham FROM product_origin po JOIN product p ON po.id = p.product_origin_id GROUP BY po.name;"
        },
        {
            "input": "Có bao nhiêu sản phẩm có nguồn gốc từ Colombia",
            "query": "SELECT COUNT(p.id) as so_luong_san_pham FROM product_origin po JOIN product p ON po.id = p.product_origin_id where po.name like '%colombia%';"
        },
        {
            "input": "Liệt kê những sản phẩm có nguồn gốc từ Colombia",
            "query": "SELECT p.name as ten_san_pham FROM product_origin po JOIN product p ON po.id = p.product_origin_id where po.name like '%colombia%';"
        },
        {
            "input": "Số lượng sản phẩm từ mỗi thương hiệu",
            "query": "SELECT b.name, COUNT(p.id) as so_luong_san_pham FROM brand b JOIN product p ON b.id = p.brand_id GROUP BY b.name;"
        },
        {
            "input": "Có bao nhiêu sản phẩm có thương hiệu Artic Fox",
            "query": "SELECT COUNT(p.id) as so_luong_san_pham FROM brand b JOIN product p ON b.id = p.brand_id where b.name like '%artic fox%';"
        },
        {
            "input": "Liệt kê những sản phẩm có thương hiệu Artic Fox",
            "query": "SELECT p.name as ten_san_pham FROM brand b JOIN product p ON b.id = p.brand_id where b.name like '%artic fox%';"
        },
        {
            "input": "Sản phẩm Red Bull Espresso Blend còn tồn kho bao nhiêu",
            "query": "select SUM(pd.stock) as so_luong_ton_kho from product p join product_detail pd on pd.product_id = p.id where p.name like '%Red Bull Espresso Blend%';"
        },
        {
            "input": "liệt kê các thương hiệu có sản phẩm đặc biệt?",
            "query": "select p.name from brand b join product p on p.brand_id = b.id where p.is_special = 1;"
        },
        {
            "input": "Thương hiệu nào có nhiều sản phẩm đặc biệt nhất?",
            "query": "SELECT b.name AS BrandName, COUNT(p.id) AS NumberOfSpecialProducts FROM brand b JOIN product p ON b.id = p.brand_id WHERE p.is_special = TRUE GROUP BY b.id, b.name ORDER BY NumberOfSpecialProducts DESC LIMIT 1;"
        },
        {
            "input": "Liệt kê những sản phẩm đặc biệt của thương hiệu StupiDucks?",
            "query": "select p.name from product p left join brand b on b.id = p.brand_id where p.is_special = 1 and b.name like '%stupiducks%';"
        },
        {
            "input": "Liệt kê những sản phẩm đặc biệt của thương hiệu StupiDucks và Artic fox?",
            "query": "select p.name from product p left join brand b on b.id = p.brand_id where p.is_special = 1 and (b.name like '%stupiducks%' or b.name like '%artic fox%');"
        },
        {
            "input": "Các loại sản phẩm không có sản phẩm nào",
            "query": "SELECT c.name FROM category c LEFT JOIN product p ON c.id = p.category_id WHERE p.id IS NULL;"
        },
        {
            "input": "Liệt kê sản phẩm thuộc loại cà phê rang",
            "query": "select p.name from product p left join category c on p.category_id = c.id where c.name like '%rang%';"
        },
        {
            "input": "Sản phẩm có giá thấp nhất trong mỗi loại sản phẩm",
            "query": "SELECT c.name AS CategoryName, p.name AS ProductName, pd.price AS Price FROM category c JOIN product p ON c.id = p.category_id JOIN product_detail pd ON p.id = pd.product_id JOIN (    SELECT p.category_id,  MIN(pd.price) AS MinPrice FROM product p JOIN product_detail pd ON p.id = pd.product_id GROUP BY p.category_id) AS MinPrices ON c.id = MinPrices.category_id AND pd.price = MinPrices.MinPrice GROUP BY c.name, p.name, pd.price ORDER BY c.name;"
        },
        {
            "input": "Trong mỗi loại sản phẩm, hãy liệt kê những loại sản phẩm được bán chạy nhất",
            "query": "WITH RankedProducts AS ( SELECT p.id, p.name AS ProductName, c.name AS CategoryName, p.sold, ROW_NUMBER() OVER(PARTITION BY p.category_id ORDER BY p.sold DESC) AS rn FROM product p JOIN category c ON p.category_id = c.id) SELECT ProductName, CategoryName, sold FROM RankedProducts WHERE rn = 1;"
        },
        {
            "input": "Sản phẩm có giá cao nhất trong mỗi loại sản phẩm",
            "query": "SELECT c.name AS CategoryName, p.name AS ProductName, pd.price AS Price FROM category c JOIN product p ON c.id = p.category_id JOIN product_detail pd ON p.id = pd.product_id JOIN (    SELECT p.category_id,  MAX(pd.price) AS MaxPrice FROM product p JOIN product_detail pd ON p.id = pd.product_id GROUP BY p.category_id) AS MaxPrices ON c.id = MaxPrices.category_id AND pd.price = MaxPrices.MaxPrice GROUP BY c.name, p.name, pd.price ORDER BY c.name;"
        },
        {
            "input": "Loại sản phẩm nào bán chạy nhất hoặc bán nhiều nhất?",
            "query": "SELECT c.name, SUM(p.sold) as tong_so_luong_ban FROM product p LEFT JOIN category c ON p.category_id = c.id GROUP BY c.id, c.name ORDER BY tong_so_luong_ban DESC limit 1;"
        },
        {
            "input": "Liệt kê các sản phẩm mới nhập hàng gần đây nhất",
            "query": "SELECT id, name, updated_at FROM product WHERE updated_at >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH) ORDER BY updated_at DESC;"
        },
        {
            "input": "Liệt kê các sản phẩm mới nhập hàng cách đây 5 tháng",
            "query": "SELECT id, name, updated_at FROM product WHERE updated_at >= DATE_SUB(CURDATE(), INTERVAL 5 MONTH) ORDER BY updated_at DESC;"
        },
        {
            "input": "Liệt kê các sản phẩm đang được giảm giá hoặc sử dụng voucher",
            "query": "(SELECT p.id, p.name, v.code FROM product p left join voucher v on p.sale_id = v.id WHERE p.sale_id IS NOT NULL) UNION (SELECT p.id, p.name, v.code FROM product p JOIN category c ON p.category_id = c.id JOIN voucher v ON c.id = v.category_id WHERE v.code IS NOT NULL AND v.expiration_date > CURRENT_DATE);"
        },
        {
            "input": "Liệt kê các sản phẩm đang được giảm giá hoặc sử dụng voucher",
            "query": "(SELECT p.id, p.name, v.code FROM product p left join voucher v on p.sale_id = v.id WHERE p.sale_id IS NOT NULL) UNION (SELECT p.id, p.name, v.code FROM product p JOIN category c ON p.category_id = c.id JOIN voucher v ON c.id = v.category_id WHERE v.code IS NOT NULL AND v.expiration_date > '2023-01-20');"
        },
        {
            "input": "Liệt kê các sản phẩm thuộc loại hạt đã rang đã được giảm giá hoặc sử dụng voucher từ ngày 20 tháng 1 năm 2023",
            "query": "(SELECT p.id, p.name, v.code FROM product p left join voucher v on p.sale_id = v.id WHERE p.sale_id IS NOT NULL) UNION (SELECT p.id, p.name, v.code FROM product p JOIN category c ON p.category_id = c.id JOIN voucher v ON c.id = v.category_id WHERE v.code IS NOT NULL AND v.expiration_date > '2023-01-20' and c.name like '%hạt đã rang%');"
        },
        {
            "input": "Tìm sản phẩm có số lượng đánh giá và số sao đánh giá trung bình cao nhất",
            "query": "SELECT p.id, p.name, COUNT(r.product_id) AS NumberOfReviews, AVG(r.rating) AS AverageRating FROM product p LEFT JOIN review r ON p.id = r.product_id GROUP BY p.id, p.name ORDER BY NumberOfReviews DESC, AverageRating DESC LIMIT 1;"
        },
        {
            "input": "Tìm sản phẩm có số lượng đánh giá và số sao đánh giá trung bình cao nhất",
            "query": "SELECT p.id, p.name, COUNT(r.product_id) AS NumberOfReviews, AVG(r.rating) AS AverageRating FROM product p LEFT JOIN review r ON p.id = r.product_id GROUP BY p.id, p.name ORDER BY AverageRating DESC, NumberOfReviews DESC LIMIT 1;"
        },
        {
            "input": "Sản phẩm mới nhất từ mỗi thương hiệu",
            "query": "SELECT b.name as BrandName, p.name as ProductName, p.created_at FROM brand b JOIN product p ON b.id = p.brand_id JOIN ( SELECT brand_id, MAX(created_at) as LatestProductDate FROM product GROUP BY brand_id ) as latest_products ON p.brand_id = latest_products.brand_id AND p.created_at = latest_products.LatestProductDate ORDER BY p.created_at DESC;"
        },
        {
            "input": "Trong mỗi loại hương vị, hãy liệt kê những sản phẩm được bán chạy nhất",
            "query": "WITH RankedProducts AS ( SELECT p.id, p.name AS ProductName, f.name AS FlavorName, p.sold, ROW_NUMBER() OVER(PARTITION BY p.flavor_id ORDER BY p.sold DESC) AS rn FROM product p    JOIN flavor f ON p.flavor_id = f.id) SELECT ProductName, FlavorName, sold FROM RankedProducts WHERE rn = 1;"
        },
        {
            "input": "Loại hương vị nào bán chạy nhất hoặc bán nhiều nhất?",
            "query": "SELECT f.name, SUM(p.sold) as tong_so_luong_ban FROM product p LEFT JOIN flavor f ON p.flavor_id = f.id GROUP BY f.id, f.name ORDER BY tong_so_luong_ban DESC limit 1;"
        },
        {
            "input": "Cà phê của bạn có những hương vị gì?",
            "query": "select f.name from flavor f;"
        },
        {
            "input": "Sản phẩm có giá thấp nhất trong mỗi loại hương vị?",
            "query": "SELECT f.name AS FlavorName, p.name AS ProductName, pd.price AS Price FROM flavor f JOIN product p ON f.id = p.flavor_id JOIN product_detail pd ON p.id = pd.product_id JOIN ( SELECT p.flavor_id,  MIN(pd.price) AS MinPrice FROM product p JOIN product_detail pd ON p.id = pd.product_id GROUP BY p.flavor_id) AS MinPrices ON f.id = MinPrices.flavor_id AND pd.price = MinPrices.MinPrice GROUP BY f.name, p.name, pd.price ORDER BY f.name;"
        },
        {
            "input": "Sản phẩm có giá cao nhất trong mỗi loại hương vị?",
            "query": "SELECT f.name AS FlavorName, p.name AS ProductName, pd.price AS Price FROM flavor f JOIN product p ON f.id = p.flavor_id JOIN product_detail pd ON p.id = pd.product_id JOIN ( SELECT p.flavor_id,  MAX(pd.price) AS MaxPrice FROM product p JOIN product_detail pd ON p.id = pd.product_id GROUP BY p.flavor_id) AS MaxPrices ON f.id = MaxPrices.flavor_id AND pd.price = MaxPrices.MaxPrice GROUP BY f.name, p.name, pd.price ORDER BY f.name;"
        },
        {
            "input": "Các sản phẩm chưa có đánh giá",
            "query": "SELECT p.name FROM product p LEFT JOIN review r ON p.id = r.product_id WHERE r.id IS NULL;"
        },
        {
            "input": "Top 3 sản phẩm có giá thấp nhất cho mỗi loại sản phẩm",
            "query": "WITH RankedProducts AS ( SELECT p.id, p.name AS ProductName, c.name AS CategoryName, pd.price, ROW_NUMBER() OVER(PARTITION BY p.category_id ORDER BY pd.price ASC) AS rn  FROM product p JOIN category c ON p.category_id = c.id JOIN product_detail pd ON p.id = pd.product_id) SELECT id, ProductName, CategoryName, price FROM RankedProducts WHERE rn <= 3;"
        },
        {
            "input": "Top 3 sản phẩm có giá cao nhất cho mỗi loại sản phẩm",
            "query": "WITH RankedProducts AS ( SELECT p.id, p.name AS ProductName, c.name AS CategoryName, pd.price, ROW_NUMBER() OVER(PARTITION BY p.category_id ORDER BY pd.price desc) AS rn  FROM product p JOIN category c ON p.category_id = c.id JOIN product_detail pd ON p.id = pd.product_id) SELECT id, ProductName, CategoryName, price FROM RankedProducts WHERE rn <= 3;"
        },

        {
            "input": "Liệt kê các sản phẩm từ thương hiệu có nhiều sản phẩm nhất",
            "query": "WITH BrandProductCount AS ( SELECT brand_id, COUNT(*) AS product_count FROM product GROUP BY brand_id), MaxProductBrand AS ( SELECT brand_id FROM BrandProductCount  WHERE product_count = (SELECT MAX(product_count) FROM BrandProductCount)) SELECT p.id, p.name AS ProductName, b.name AS BrandName FROM  product p   JOIN brand b ON p.brand_id = b.id JOIN MaxProductBrand mpb ON b.id = mpb.brand_id;"
        },
        {
            "input": "Tổng số lượng sản phẩm có trong mỗi nguồn gốc",
            "query": "SELECT po.name, COUNT(p.id) FROM product_origin po JOIN product p ON po.id = p.product_origin_id GROUP BY po.name;"
        },
        {
            "input": "Cà phê của bạn có nguồn gốc như thế nào",
            "query": "select po.name from product_origin po;"
        },
        {
            "input": "Loại cà phê thuộc nguồn gốc nào bán chạy nhất hoặc bán nhiều nhất",
            "query": "SELECT po.name, SUM(p.sold) as tong_so_luong_ban FROM product p LEFT JOIN product_origin po ON p.product_origin_id = po.id GROUP BY po.id, po.name ORDER BY tong_so_luong_ban DESC limit 1;"
        },
        {
            "input": "Với mỗi nguồn gốc, đã bán được bao nhiêu sản phẩm",
            "query": "SELECT po.name, SUM(p.sold) as tong_so_luong_ban FROM product p LEFT JOIN product_origin po ON p.product_origin_id = po.id GROUP BY po.id, po.name ORDER BY tong_so_luong_ban;"
        },
        {
            "input": "Trong mỗi nguồn gốc, hãy liệt kê những sản phẩm được bán chạy nhất",
            "query": "WITH RankedProducts AS ( SELECT p.id, p.name AS ProductName, f.name AS FlavorName, p.sold, ROW_NUMBER() OVER(PARTITION BY p.flavor_id ORDER BY p.sold DESC) AS rn FROM product p    JOIN flavor f ON p.flavor_id = f.id) SELECT ProductName, FlavorName, sold FROM RankedProducts WHERE rn = 1;"
        },
        {
            "input": "Sản phẩm có giá thấp nhất trong mỗi nguồn gốc, xuất xứ",
            "query": "SELECT po.name AS ProductOriginName, p.name AS ProductName, pd.price AS Price FROM product_origin po JOIN product p ON po.id = p.product_origin_id JOIN product_detail pd ON p.id = pd.product_id JOIN ( SELECT p.product_origin_id,  MIN(pd.price) AS MinPrice FROM product p JOIN product_detail pd ON p.id = pd.product_id GROUP BY p.product_origin_id) AS MinPrices ON po.id = MinPrices.product_origin_id AND pd.price = MinPrices.MinPrice GROUP BY po.name, p.name, pd.price ORDER BY po.name;"
        },
        {
            "input": "Sản phẩm có giá cao nhất trong mỗi nguồn gốc, xuất xứ",
            "query": "SELECT po.name AS ProductOriginName, p.name AS ProductName, pd.price AS Price FROM product_origin po JOIN product p ON po.id = p.product_origin_id JOIN product_detail pd ON p.id = pd.product_id JOIN ( SELECT p.product_origin_id,  MAX(pd.price) AS MaxPrice FROM product p JOIN product_detail pd ON p.id = pd.product_id GROUP BY p.product_origin_id) AS MaxPrices ON po.id = MaxPrices.product_origin_id AND pd.price = MaxPrices.MaxPrice GROUP BY po.name, p.name, pd.price ORDER BY po.name;"
        },
        {
            "input": "Các sản phẩm có tồn kho dưới 5 mà không phải là sản phẩm giới hạn",
            "query": "SELECT p.name FROM product p JOIN product_detail pd ON p.id = pd.product_id WHERE pd.stock < 5 AND p.is_limited = false;"
        },
        {
            "input": "Liệt kê các sản phẩm được bán với voucher giảm giá cao nhất",
            "query": "SELECT p.id AS product_id, p.name AS product_name, c.name AS category_name, v.name AS voucher_name, MAX(v.discount) AS max_discount FROM product p LEFT JOIN category c ON p.category_id = c.id LEFT JOIN voucher v ON p.sale_id = v.id OR (p.sale_id IS NULL AND p.category_id = v.category_id) where v.discount is not null GROUP BY  p.id, p.name, c.name, v.name ORDER BY  max_discount DESC;"
        },
        {
            "input": "Hiện tại, liệt kê 10 sản phẩm được bán với voucher giảm giá cao nhất",
            "query": "SELECT p.id AS product_id, p.name AS product_name, c.name AS category_name, v.name AS voucher_name, MAX(v.discount) AS max_discount FROM product p LEFT JOIN category c ON p.category_id = c.id LEFT JOIN voucher v ON p.sale_id = v.id OR (p.sale_id IS NULL AND p.category_id = v.category_id) where v.discount is not null and v.expiration_date > CURRENT_DATE GROUP BY p.id, p.name, c.name, v.name ORDER BY max_discount DESC;"
        },
        {
            "input": "Top 5 sản phẩm có tồn kho cao nhất",
            "query": "select p.name, SUM(pd.stock) from product_detail pd left join product p on pd.product_id = p.id group by pd.product_id order by sum(pd.stock) desc limit 5;"
        },
        {
            "input": "Các loại sản phẩm không có trong kho",
            "query": "SELECT c.name FROM category c JOIN product p ON c.id = p.category_id JOIN product_detail pd ON p.id = pd.product_id WHERE pd.stock = 0 GROUP BY c.name;"
        },
        {
            "input": "Những sản phẩm có đánh giá trung bình dưới 4 sao",
            "query": "SELECT p.name, AVG(r.rating) as so_sao_trung_binh FROM product p JOIN review r ON p.id = r.product_id GROUP BY p.id HAVING AVG(r.rating) < 4;"
        },
        {
            "input": "Sản phẩm từ thương hiệu không còn hoạt động",
            "query": "SELECT p.name FROM product p JOIN brand b ON p.brand_id = b.id WHERE b.status = 0;"
        },
        {
            "input": "Sản phẩm từ thương hiệu còn hoạt động",
            "query": "SELECT p.name FROM product p JOIN brand b ON p.brand_id = b.id WHERE b.status = 1;"
        },
        {
            "input": "Các sản phẩm mới được thêm vào trong tuần qua",
            "query": "SELECT name FROM product WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 WEEK);"
        },
        {
            "input": "Các sản phẩm dự kiến sẽ ra mắt trong tháng tới",
            "query": "SELECT id, name, created_at FROM product WHERE created_at BETWEEN NOW() AND DATE_ADD(NOW(), INTERVAL 1 MONTH);"
        },
        {
            "input": "Các sản phẩm đã ra mắt trong 1 tháng trước",
            "query": "SELECT id, name, created_at FROM product WHERE created_at BETWEEN DATE_SUB(NOW(), INTERVAL 1 MONTH) AND NOW();"
        },
        {
            "input": "Sản phẩm từ nhà cung cấp đã bị dừng hợp tác",
            "query": "SELECT p.name FROM product p JOIN vendor v ON p.vendor_id = v.id WHERE v.status = false;"
        },
        {
            "input": "Các loại sản phẩm có sản phẩm giới hạn",
            "query": "SELECT DISTINCT c.name, p.name FROM category c JOIN product p ON c.id = p.category_id WHERE p.is_limited = true;"
        },
        {
            "input": "Sản phẩm có số lượng bán ra thấp nhất",
            "query": "SELECT p.name, p.sold FROM product p ORDER BY p.sold ASC LIMIT 1;;"
        },
        {
            "input": "Top 3 thương hiệu có đánh giá cao nhất",
            "query": "SELECT b.name, AVG(r.rating) FROM brand b JOIN product p ON b.id = p.brand_id JOIN review r ON p.id = r.product_id GROUP BY b.id ORDER BY AVG(r.rating) DESC LIMIT 3;"
        },
        {
            "input": "Các sản phẩm có trong cả hương vị trái cây và rượu nho",
            "query": "select * from product p left join flavor f on p.flavor_id = f.id where f.name like '%trái cây%' and f.name like '%rượu nho%';"
        },
        {
            "input": "Các sản phẩm có hương vị trái cây và rượu nho",
            "query": "select * from product p left join flavor f on p.flavor_id = f.id where f.name like '%trái cây%' or f.name like '%rượu nho%';"
        }
    ]
    system_prefix = """You are an AI friendly and helpful AI assistant for question-answering user's task about 
    products, orders, user information, .... Given an input question, create a syntactically correct {dialect} query 
    to run, then look at the results of the query and return the answer as a friendly, conversational retail shopping 
    assistant. Unless the user specifies a specific number of examples they wish to obtain, always limit your query 
    to at most {top_k} results. You can order the results by a relevant column to return the most interesting 
    examples in the database. Never query for all the columns from a specific table, only ask for the relevant 
    columns given the question. You have access to tools for interacting with the database. Only use the given tools. 
    Only use the information returned by the tools to construct your final answer. You MUST double check your query 
    before executing it. If you get an error while executing a query, rewrite the query and try again.

    Please do follow up these rules before executing query:

    1. DO NOT make up data and no yapping.

    2. DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    3. If the question is about describing or list all tables in database, DO NOT do that. Instead, respond politely that you cannot disclose information about the database schema for security reasons.

    4. If the question is about listing all user's information, DO NOT do that.

    5. If the question is about user then ask them their username and email before querying data

    6. If the question is about order, ask them their order code and phone number before querying data.

    7. If the question is about the store's finances or related things.

    8. If the question does not seem related to the database, just say you don't know as the answer.

    Here is the relevant table info: {table_info}
    Here are some examples of user inputs and their corresponding SQL queries:"""
    system_suffix = """
    Begin!

    Relevant pieces of previous conversation:
    {history}
    (You do not need to use these pieces of information if not relevant)
    """

    def __init__(self, db_uri, memory):
        self.db_uri = db_uri
        self.db = SQLDatabase.from_uri(db_uri, sample_rows_in_table_info=3)
        self.memory = memory

    def create_agent(self):
        llm = ChatOpenAI(model=self.model, temperature=0)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
        vector_store = Chroma()
        vector_store.delete_collection()
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            self.examples,
            OpenAIEmbeddings(),
            vector_store,
            k=5,
            input_keys=["input"]
        )
        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=PromptTemplate.from_template(
                "User input: {input}\nSQL query: {query}"
            ),
            input_variables=["input", "dialect", "top_k", "table_info"],
            prefix=self.system_prefix,
            suffix=self.system_suffix,
        )
        full_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=few_shot_prompt),
                MessagesPlaceholder("history", optional=True),
                MessagesPlaceholder("agent_scratchpad"),
                ("human", "{input}"),
            ]
        )

        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            prompt=full_prompt,
            verbose=True,
            agent_type="openai-tools",
            agent_executor_kwargs={'memory': self.memory}
        )
        return agent

    def run(self, input_text):
        agent = self.create_agent()
        response = agent.invoke({"input": input_text})
        return response
