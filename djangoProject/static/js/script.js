$(document).ready(function () {
    var dropZone = $('#upload-container');

    $('#file-input').focus(function () {
        $('label').addClass('focus');
    })
        .focusout(function () {
            $('label').removeClass('focus');
        });


    dropZone.on('drag dragstart dragend dragover dragenter dragleave drop', function () {
        return false;
    });

    dropZone.on('dragover dragenter', function () {
        dropZone.addClass('dragover');
        $(".switcher .item.active").addClass('dragover')
    });

    dropZone.on('dragleave', function (e) {
        let dx = e.pageX - dropZone.offset().left;
        let dy = e.pageY - dropZone.offset().top;
        if ((dx < 0) || (dx > dropZone.width()) || (dy < 0) || (dy > dropZone.height())) {
            dropZone.removeClass('dragover');
            $(".switcher .item.active").removeClass('dragover');
        }
    });

    dropZone.on('drop', function (e) {
        dropZone.removeClass('dragover');
        let files = e.originalEvent.dataTransfer.files;
        sendFiles(files);
    });

    $('#file-input').change(function () {
        let files = this.files;
        sendFiles(files);
    });


    function sendFiles(files) {
        let maxFileSize = 5242880;
        let Data = new FormData();
        $(files).each(function (index, file) {
            Data.append('file', file);
        });

        var token = $('input[name="csrfmiddlewaretoken"]').attr('value');
        Data.append('csrfmiddlewaretoken', token);

        $(".container").css("display", "none");
        $(".loading").css("display", "flex");

        var form = document.getElementById("upload-container");
        form.submit();
        // $.ajax({
        //     url: dropZone.attr('action'),
        //     type: dropZone.attr('method'),
        //     data: Data,
        //     contentType: false,
        //     processData: false,
        //     success: function (data) {
        //         $(".loading").css("display", "none");
        //         $("#myCanvas").attr("src", data);
        //         $("#myCanvas").css("display", "flex");
        //         // alert(data);
        //         // alert('Файлы были успешно загружены!');
        //     }
        // });
    }
})