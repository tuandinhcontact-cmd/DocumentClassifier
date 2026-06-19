document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('classify-form');
    const headlineInput = document.getElementById('headline');
    const descInput = document.getElementById('short_description');
    const btnSubmit = document.getElementById('btn-submit');
    const loader = document.getElementById('loader');
    
    const emptyCard = document.getElementById('empty-card');
    const resultCard = document.getElementById('result-card');
    const timelineCard = document.getElementById('timeline-card');
    
    const categoryBadge = document.getElementById('category-badge');
    const cleanedTextVal = document.getElementById('cleaned-text-val');
    const timelineContainer = document.getElementById('timeline');
    
    // 1. Xử lý sự kiện click vào các mẫu tin thử nhanh
    const sampleBtns = document.querySelectorAll('.sample-btn');
    sampleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            headlineInput.value = btn.getAttribute('data-headline');
            descInput.value = btn.getAttribute('data-desc');
            
            // Tự động cuộn đến phần nhập liệu và submit
            headlineInput.scrollIntoView({ behavior: 'smooth' });
            form.dispatchEvent(new Event('submit'));
        });
    });
    
    // 2. Submit form phân loại
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const headline = headlineInput.value.trim();
        const short_description = descInput.value.trim();
        
        if (!headline || !short_description) return;
        
        // Hiển thị trạng thái Loading
        btnSubmit.disabled = true;
        loader.style.display = 'block';
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ headline, short_description })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                renderResult(result);
            } else {
                alert('Lỗi: ' + (result.error || 'Có lỗi xảy ra trên server.'));
            }
        } catch (error) {
            console.error(error);
            alert('Lỗi kết nối tới server Flask! Hãy chắc chắn server đang chạy.');
        } finally {
            // Tắt trạng thái Loading
            btnSubmit.disabled = false;
            loader.style.display = 'none';
        }
    });
    
    // 3. Hàm hiển thị kết quả lên giao diện
    function renderResult(data) {
        // Ẩn Empty Card, hiển thị Result & Timeline Cards
        emptyCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
        timelineCard.classList.remove('hidden');
        
        // Cập nhật nhãn kết quả và text sạch
        categoryBadge.innerText = data.prediction;
        cleanedTextVal.innerText = data.cleaned_text;
        
        // Reset container timeline
        timelineContainer.innerHTML = '';
        
        // Tạo timeline động với hiệu ứng trễ (stagger animation)
        data.cascade_path.forEach((item, index) => {
            const timelineItem = document.createElement('div');
            
            // Xác định class style dựa trên trạng thái
            let stateClass = 'passed';
            if (item.matched) {
                stateClass = item.status.includes('Default') ? 'default' : 'matched';
            }
            
            timelineItem.className = `timeline-item ${stateClass}`;
            // Thêm hiệu ứng trễ cho từng phần tử xuất hiện tuần tự
            timelineItem.style.animationDelay = `${index * 0.15}s`;
            
            // Format số phần trăm xác suất
            const probPercent = (item.probability * 100).toFixed(1);
            
            timelineItem.innerHTML = `
                <div class="timeline-marker"></div>
                <div class="timeline-content">
                    <div class="timeline-details">
                        <h4>Bước ${item.stage}: Phân lớp "${item.class_name}"</h4>
                        <p>
                            Trạng thái: <strong>${item.status}</strong>
                            ${item.feature_type && item.feature_type !== 'N/A' ? `
                            <span class="feat-badge ${item.feature_type.toLowerCase()}">${item.feature_type.toUpperCase()}</span>
                            <span class="size-info">(${item.class_size.toLocaleString()} mẫu)</span>
                            ` : ''}
                        </p>
                    </div>
                    <div class="timeline-stats">
                        <div class="prob-val">${probPercent}%</div>
                        <div class="probability-bar-container">
                            <div class="probability-bar" style="width: ${probPercent}%"></div>
                        </div>
                    </div>
                </div>
            `;
            
            timelineContainer.appendChild(timelineItem);
        });
    }
});
