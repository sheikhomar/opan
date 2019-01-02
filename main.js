var coll = document.getElementsByClassName("collapsible");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}

var openAll = document.getElementsByClassName('toggle-all')[0];
openAll.addEventListener('click', function() {
  for (i = 0; i < coll.length; i++) {
    coll[i].click();
  }
});