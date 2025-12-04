import qrcode

# Your website URL
url = "https://example.com"  # <- replace with your actual website

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create an image
img = qr.make_image(fill_color="black", back_color="white")

# Save as PNG
img.save("fitness_qrcode.png")
print("QR code saved as fitness_qrcode.png")
