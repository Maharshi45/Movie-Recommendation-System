"use strict";
new WOW().init();
$(window).load(function() { $(".bg_load").fadeOut("slow") })

function openNav() {
    document.getElementById("mySidenav").style.width = "250px";
    document.getElementById("main").style.marginLeft = "250px"
}

function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
    document.getElementById("main").style.marginLeft = "0"
}
// $('.select2').	select2();
$("#quick_search").on('input', function(event) {
    var value = $(this).val();
    if (value.length > 2) {
        $.get('/quicksearch/' + value, function(data) {
            $('.search_results').empty();
            if (data.movies && data.movies.length > 0) {
                $('.search_results').show();
                for (var i = 0; i < data.movies.length; i++) {
                    var m = data.movies[i];
                    $('.search_results').append('<li><a href="/movie/' + m.slug + '"><img src="' + m.image_cover + '"><div><span>' + m.title + '</span><p>' + m.year + '</p></div></a></li>')
                }
            } else { $('.search_results').hide() }
        })
    }
});
$("body,html").on('click', function(event) { if ($('.search_results').css('display') == "block") { $('.search_results').hide() } });
$('#btn-generate-rss').click(function(event) {
    var baseurl = "https://yts.ws/rss";
    var name = $('#input_movie_name').val();
    var quality = $('#select_quality').find(":selected").text();
    var genre = $('#select_genre').find(":selected").text();
    var rating = $('#select_rating').find(":selected").text();
    if (name == "") { name = "0" }
    if (quality == "All") { quality = "all" }
    if (genre == "All") { genre = "all" }
    if (rating == "All") { rating = "0" }
    var url = baseurl + "/" + name + "/" + quality + "/" + genre + "/" + rating;
    $('.generate-container').show();
    $('#input_generated').val(url)
});

$('.comment_edit').click(function(event) {
    var target = $(this).attr('data-target');
    $('#edit_comment_' + target).show();
    $(this).hide();
    $('#comment_save_' + target).show();
    $('.comment_reply').hide();
});

$('.comment_save').click(function(event) {
    var target = $(this).attr('data-target');
    if ($(this).attr('id').indexOf('reply') > -1) {
        $('#reply_comment_' + target + ' > form').submit()
    } else {
        $('#edit_comment_' + target + ' > form').submit()
    }
});

$('.comment_delete').click(function(event) {
    var target = $(this).attr('data-target');
    $('#delete_comment_' + target + ' > form').submit()
});

$('.comment_reply').click(function(event) {
    var target = $(this).attr('data-target');
    $('#reply_comment_' + target).show();
    $(this).hide();
    $('#comment_reply_' + target).show();
    $('.comment_edit').hide();
})