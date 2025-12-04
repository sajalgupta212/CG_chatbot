-- Sample SQL Dump for RAG / Text Retrieval Testing
-- Database: ecommerce_app
-- Author: Demo
-- Created: 2025-01-01

DROP DATABASE IF EXISTS ecommerce_app;
CREATE DATABASE ecommerce_app;
USE ecommerce_app;

-- -------------------------
-- TABLES
-- -------------------------

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL
);

CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    stock INT DEFAULT 0,
    category_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL
);

CREATE TABLE categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    parent_id INT DEFAULT NULL
);

CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL,
    status VARCHAR(30) DEFAULT 'PENDING',
    total DECIMAL(12,2) NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(id),
    FOREIGN KEY(product_id) REFERENCES products(id)
);

-- Inventory history
CREATE TABLE inventory_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    action VARCHAR(20) NOT NULL,   -- ADD, REMOVE, SALE, RESTOCK
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -------------------------
-- SAMPLE DATA
-- -------------------------

INSERT INTO categories (name) VALUES
('Books'),
('Electronics'),
('Home'),
('Fitness'),
('Stationery'),
('Clothing'),
('Shoes'),
('Mobile Phones'),
('Laptops'),
('Furniture');

INSERT INTO users (username, email, password_hash) VALUES
('johnsmith', 'john@example.com','fake_hash1'),
('janedoe', 'jane@example.com','fake_hash2'),
('bob', 'bob@example.com','fake_hash3'),
('alice', 'alice@example.com','fake_hash4'),
('admin', 'admin@example.com','fake_hash5');

INSERT INTO products (name, description, price, stock, category_id) VALUES
('Running Shoes', 'High-quality running shoes', 59.99, 100, 7),
('Laptop XYZ', '14-inch lightweight laptop', 999.99, 25, 9),
('Standing Desk', 'Adjustable height standing desk', 299.99, 40, 10),
('Water Bottle', 'Insulated stainless-steel bottle', 19.99, 80, 4),
('Backpack', 'Large waterproof backpack', 49.99, 60, 6),
('Headphones', 'Noise-cancelling over-ear', 199.99, 30, 2),
('Novel Book', 'Mystery / thriller', 9.99, 200, 1);

INSERT INTO orders (user_id, total, status) VALUES
(1, 129.98, 'COMPLETED'),
(3, 49.99, 'PENDING'),
(2, 59.99, 'PENDING'),
(5, 199.99, 'CANCELLED'),
(4, 999.99, 'COMPLETED');

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 2, 59.99),
(2, 5, 1, 49.99),
(3, 4, 3, 19.99),
(4, 6, 1, 199.99),
(5, 2, 1, 999.99);

-- Insert many logs
INSERT INTO inventory_log (product_id, quantity, action) VALUES
(1, 100, 'RESTOCK'),
(1, 2, 'SALE'),
(1, 20, 'RESTOCK'),
(4, 5, 'SALE'),
(2, 1, 'SALE'),
(2, 10, 'RESTOCK'),
(2, 2, 'SALE'),
(3, 5, 'RESTOCK'),
(3, 5, 'SALE'),
(3, 5, 'RESTOCK'),
(3, 3, 'SALE'),
(5, 10, 'RESTOCK'),
(6, 1, 'SALE'),
(7, 15, 'RESTOCK'),
(7, 5, 'SALE');

-- +++++ Repeat log records to make file large
INSERT INTO inventory_log (product_id, quantity, action)
VALUES
(1, 1, 'SALE'),
(2, 3, 'SALE'),
(3, 1, 'RESTOCK'),
(4, 2, 'RESTOCK'),
(5, 1, 'SALE'),
(1, 3, 'SALE'),
(2, 5, 'RESTOCK'),
(3, 8, 'SALE'),
(4, 6, 'RESTOCK'),
(5, 6, 'SALE'),
(6, 6, 'RESTOCK'),
(7, 4, 'SALE'),
(8, 3, 'RESTOCK'),
(9, 5, 'SALE'),
(10, 9, 'RESTOCK');

-- -------------------------
-- VIEWS
-- -------------------------

CREATE VIEW view_user_orders AS
SELECT u.username, o.id AS order_id, o.status, o.total, o.created_at
FROM users u
JOIN orders o ON u.id = o.user_id;

CREATE VIEW view_stock AS
SELECT p.id, p.name, p.stock, c.name AS category
FROM products p
JOIN categories c ON c.id = p.category_id;

-- -------------------------
-- PROCEDURES
-- -------------------------

DELIMITER $$

CREATE PROCEDURE add_inventory_log(
    IN p_product_id INT,
    IN p_qty INT,
    IN p_action VARCHAR(20)
)
BEGIN
    INSERT INTO inventory_log (product_id, quantity, action) 
    VALUES (p_product_id, p_qty, p_action);
END $$

CREATE PROCEDURE create_order(
    IN userId INT,
    IN productId INT,
    IN qty INT
)
BEGIN
    DECLARE price DECIMAL(10,2);

    SELECT price INTO price FROM products WHERE id = productId;

    INSERT INTO orders(user_id, total, status) VALUES (userId, price * qty, 'PENDING');
    INSERT INTO order_items(order_id, product_id, quantity, unit_price)
    VALUES (LAST_INSERT_ID(), productId, qty, price);
END $$

DELIMITER ;

-- -------------------------
-- TRIGGERS
-- -------------------------

DELIMITER $$

CREATE TRIGGER trg_reduce_stock AFTER INSERT ON order_items
FOR EACH ROW
BEGIN
    UPDATE products
    SET stock = stock - NEW.quantity
    WHERE id = NEW.product_id;

    INSERT INTO inventory_log (product_id, quantity, action)
    VALUES (NEW.product_id, NEW.quantity, 'SALE');
END $$

DELIMITER ;

-- END
