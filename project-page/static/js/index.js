(function () {
  "use strict";

  const ratioResults = {
    2: "static/images/ratio/ratio-02.jpg",
    3: "static/images/ratio/ratio-03.jpg",
    4: "static/images/ratio/ratio-04.jpg",
    5: "static/images/ratio/ratio-05.jpg",
    6: "static/images/ratio/ratio-06.jpg",
    7: "static/images/ratio/ratio-07.jpg",
    8: "static/images/ratio/ratio-08.jpg",
    9: "static/images/ratio/ratio-09.jpg"
  };

  const selectiveExamples = {
    "low-snow": {
      inputLabel: "Low + Snow",
      removeLabel: "Remove Low-light",
      preserveLabel: "Preserve Snow",
      summary: "Selective low-light enhancement should retain the snow component.",
      portrait: false,
      images: {
        input: "static/images/selective/low-snow-input.jpg",
        baseline: "static/images/selective/low-snow-onerestore.jpg",
        cure: "static/images/selective/low-snow-cure.jpg",
        target: "static/images/selective/low-snow-target.jpg"
      },
      alt: {
        input: "Input image affected by low light and snow",
        baseline: "OneRestore result for selective low-light removal",
        cure: "CURE result removing low light while preserving snow",
        target: "Target image with snow preserved"
      }
    },
    "haze-rain": {
      inputLabel: "Haze + Rain",
      removeLabel: "Remove Haze",
      preserveLabel: "Preserve Rain",
      summary: "Selective dehazing should retain the rain component instead of clearing both.",
      portrait: true,
      images: {
        input: "static/images/selective/haze-rain-input.jpg",
        baseline: "static/images/selective/haze-rain-onerestore.jpg",
        cure: "static/images/selective/haze-rain-cure.jpg",
        target: "static/images/selective/haze-rain-target.jpg"
      },
      alt: {
        input: "Input image affected by haze and rain",
        baseline: "OneRestore result for selective haze removal",
        cure: "CURE result removing haze while preserving rain",
        target: "Target image with rain preserved"
      }
    }
  };

  const orderExamples = {
    onerestore: {
      first: "static/images/order/onerestore-low-snow.jpg",
      second: "static/images/order/onerestore-snow-low.jpg",
      metric: "1.02 dB",
      caption: "OneRestore changes noticeably when the degradation removal order is reversed."
    },
    cure: {
      first: "static/images/order/cure-low-snow.jpg",
      second: "static/images/order/cure-snow-low.jpg",
      metric: "0.02 dB",
      caption: "CURE produces closely aligned outputs across both processing orders."
    }
  };

  const supplementaryFigures = [
    {
      number: 5,
      category: "identity",
      categoryLabel: "Identity preservation",
      title: "Identity operation on snow",
      caption: "When snow is meant to remain, the identity embedding preserves the degraded input instead of over-restoring it.",
      tags: ["Identity embedding", "Snow preservation"],
      alt: "Supplementary identity-operation study on snow images"
    },
    {
      number: 6,
      category: "identity",
      categoryLabel: "Identity preservation",
      title: "Identity under composite degradation",
      caption: "Identity behavior remains stable for a Low + Haze composite input, validating preservation in a harder setting.",
      tags: ["Identity embedding", "Low + Haze"],
      alt: "Supplementary identity-operation study on low-light and haze composites"
    },
    {
      number: 7,
      category: "ratio",
      categoryLabel: "Ratio control",
      title: "Continuous control on composite inputs",
      caption: "Restoration trajectories across mixing ratios for Haze + Snow and Low + Rain show smooth, degradation-specific control.",
      tags: ["Ratio control", "Double degradation"],
      alt: "Supplementary ratio-control trajectories for haze and snow and low-light and rain"
    },
    {
      number: 8,
      category: "ratio",
      categoryLabel: "Ratio control",
      title: "Ratio control with triple degradations",
      caption: "Continuous adjustment extends to Low + Haze + Snow and Low + Haze + Rain, including more complex unseen ratios.",
      tags: ["Ratio control", "Triple degradation"],
      alt: "Supplementary ratio-control results for triple composite degradations"
    },
    {
      number: 9,
      category: "ratio",
      categoryLabel: "Ratio control",
      title: "Digital degradation control",
      caption: "The same ratio mechanism controls restoration strength for Blur and Blur + JPEG artifacts.",
      tags: ["Ratio control", "Blur · JPEG"],
      alt: "Supplementary ratio-control results for blur and JPEG artifacts"
    },
    {
      number: 10,
      category: "selective",
      categoryLabel: "Selective control",
      title: "Target one factor in a pair",
      caption: "CURE selectively removes haze from Haze + Snow or rain from Low + Rain while preserving the other factor.",
      tags: ["Selective restoration", "Haze · Rain"],
      alt: "Supplementary selective-restoration results targeting haze and rain"
    },
    {
      number: 11,
      category: "selective",
      categoryLabel: "Selective control",
      title: "Preserve the coupled degradation",
      caption: "Selective low-light enhancement and desnowing retain the non-target degradation inside composite inputs.",
      tags: ["Selective restoration", "Low-light · Snow"],
      alt: "Supplementary selective-restoration results targeting low-light and snow"
    },
    {
      number: 12,
      category: "selective",
      categoryLabel: "Selective control",
      title: "Selectivity for digital artifacts",
      caption: "The disentangled representation also supports component-wise restoration for Blur + Noise and Noise + JPEG.",
      tags: ["Selective restoration", "Blur · Noise · JPEG"],
      alt: "Supplementary selective-restoration results for digital degradation composites"
    },
    {
      number: 13,
      category: "selective",
      categoryLabel: "Selective control",
      title: "Select what—and how much",
      caption: "Selective restoration and continuous ratio control work together, exposing both control dimensions in one model.",
      tags: ["Joint control", "Selection + intensity"],
      alt: "Supplementary results combining selective restoration and ratio control"
    },
    {
      number: 14,
      category: "order",
      categoryLabel: "Order dependency",
      title: "Low + Snow restoration order",
      caption: "One- and two-stage Low + Snow restoration remain consistent when the requested processing order changes.",
      tags: ["Order invariance", "Low + Snow"],
      alt: "Supplementary restoration-order study for low-light and snow"
    },
    {
      number: 15,
      category: "order",
      categoryLabel: "Order dependency",
      title: "Haze + Rain restoration order",
      caption: "Reversing dehazing and deraining exposes the order sensitivity of prior methods and CURE's consistency.",
      tags: ["Order invariance", "Haze + Rain"],
      alt: "Supplementary restoration-order study for haze and rain"
    },
    {
      number: 16,
      category: "order",
      categoryLabel: "Order dependency",
      title: "Blur + JPEG restoration order",
      caption: "Order-invariant behavior carries over from weather corruptions to mixed digital artifacts.",
      tags: ["Order invariance", "Blur + JPEG"],
      alt: "Supplementary restoration-order study for blur and JPEG artifacts"
    }
  ].map(function (figure) {
    figure.src = "static/images/supplement/figure-" + String(figure.number).padStart(2, "0") + ".jpg";
    return figure;
  });

  function preloadImages(paths) {
    paths.forEach(function (path) {
      const image = new Image();
      image.src = path;
    });
  }

  function swapImage(element, source, alt) {
    if (!element || element.getAttribute("src") === source) {
      if (element && alt) element.alt = alt;
      return;
    }

    const container = element.closest("figure") || element;
    container.classList.add("is-changing");
    element.onload = function () {
      container.classList.remove("is-changing");
      element.onload = null;
    };
    element.onerror = function () {
      container.classList.remove("is-changing");
      element.onerror = null;
    };
    element.src = source;
    if (alt) element.alt = alt;
  }

  function setupRatioControl() {
    const slider = document.getElementById("ratio-slider");
    const image = document.getElementById("ratio-image");
    const valueLabel = document.getElementById("ratio-value");
    const stateLabel = document.getElementById("ratio-state");
    const description = document.getElementById("ratio-description");

    if (!slider || !image) return;

    function updateRatio() {
      const step = Number(slider.value);
      const ratio = (step / 10).toFixed(1);
      const progress = ((step - Number(slider.min)) / (Number(slider.max) - Number(slider.min))) * 100;

      image.classList.add("is-changing");
      image.onload = function () {
        image.classList.remove("is-changing");
        image.onload = null;
      };
      image.src = ratioResults[step];
      image.alt = "Snow restoration result at intensity ratio " + ratio;
      slider.style.setProperty("--ratio-progress", progress + "%");
      slider.setAttribute("aria-valuenow", ratio);
      slider.setAttribute("aria-valuetext", "Restoration ratio " + ratio);
      valueLabel.textContent = "w = " + ratio;

      if (step <= 3) {
        stateLabel.textContent = "Preserve more";
        description.textContent = "A low ratio keeps more of the original degradation and stays closer to the identity operation.";
      } else if (step <= 6) {
        stateLabel.textContent = "Partial restoration";
        description.textContent = "The model removes part of the target degradation while retaining a controlled amount.";
      } else {
        stateLabel.textContent = "Restore more";
        description.textContent = "A high ratio strongly removes the target degradation and approaches full restoration.";
      }
    }

    slider.addEventListener("input", updateRatio);
    updateRatio();
    preloadImages(Object.values(ratioResults));
  }

  function setupSelectiveExamples() {
    const buttons = document.querySelectorAll("[data-selective-example]");
    const inputChip = document.getElementById("selective-input-chip");
    const removeChip = document.getElementById("selective-remove-chip");
    const preserveChip = document.getElementById("selective-preserve-chip");
    const summary = document.getElementById("selective-summary");
    const images = {
      input: document.getElementById("selective-input"),
      baseline: document.getElementById("selective-baseline"),
      cure: document.getElementById("selective-cure"),
      target: document.getElementById("selective-target")
    };

    function selectExample(key) {
      const example = selectiveExamples[key];
      if (!example) return;

      buttons.forEach(function (button) {
        const active = button.dataset.selectiveExample === key;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });

      inputChip.textContent = example.inputLabel;
      removeChip.innerHTML = '<i class="fas fa-minus" aria-hidden="true"></i> ' + example.removeLabel;
      preserveChip.innerHTML = '<i class="fas fa-shield-alt" aria-hidden="true"></i> ' + example.preserveLabel;
      summary.textContent = example.summary;

      Object.keys(images).forEach(function (name) {
        const card = images[name].closest("figure");
        card.classList.toggle("is-portrait", example.portrait);
        swapImage(images[name], example.images[name], example.alt[name]);
      });
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        selectExample(button.dataset.selectiveExample);
      });
    });

    Object.values(selectiveExamples).forEach(function (example) {
      preloadImages(Object.values(example.images));
    });
  }

  function setupOrderComparison() {
    const buttons = document.querySelectorAll("[data-order-method]");
    const first = document.getElementById("order-first");
    const second = document.getElementById("order-second");
    const metric = document.getElementById("order-metric-value");
    const caption = document.getElementById("order-caption");

    function selectMethod(method) {
      const example = orderExamples[method];
      if (!example) return;

      buttons.forEach(function (button) {
        const active = button.dataset.orderMethod === method;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });

      const displayName = method === "cure" ? "CURE" : "OneRestore";
      swapImage(first, example.first, displayName + " result after low-light enhancement followed by desnowing");
      swapImage(second, example.second, displayName + " result after desnowing followed by low-light enhancement");
      metric.textContent = example.metric;
      caption.textContent = example.caption;
    }

    buttons.forEach(function (button) {
      button.addEventListener("click", function () {
        selectMethod(button.dataset.orderMethod);
      });
    });

    Object.values(orderExamples).forEach(function (example) {
      preloadImages([example.first, example.second]);
    });
  }

  function setupSupplementExplorer() {
    const section = document.getElementById("supplement");
    const categoryButtons = Array.from(document.querySelectorAll("[data-supp-category]"));
    const jumpButtons = Array.from(document.querySelectorAll("[data-supp-jump]"));
    const filmstrip = document.getElementById("supp-filmstrip");
    const mainImage = document.getElementById("supp-main-image");
    const imageButton = document.getElementById("supp-image-button");
    const categoryLabel = document.getElementById("supp-category-label");
    const counter = document.getElementById("supp-counter");
    const figureNumber = document.getElementById("supp-figure-number");
    const title = document.getElementById("supp-title");
    const figcaptionLabel = document.getElementById("supp-figcaption-label");
    const caption = document.getElementById("supp-caption");
    const tags = document.getElementById("supp-tags");
    const previousButton = document.getElementById("supp-prev");
    const nextButton = document.getElementById("supp-next");
    const openButton = document.getElementById("supp-open");
    const lightbox = document.getElementById("supp-lightbox");
    const lightboxImage = document.getElementById("supp-lightbox-image");
    const lightboxNumber = document.getElementById("supp-lightbox-number");
    const lightboxTitle = document.getElementById("supp-lightbox-title");
    const lightboxClose = document.getElementById("supp-lightbox-close");
    const lightboxBackdrop = lightbox ? lightbox.querySelector("[data-lightbox-close]") : null;

    if (!section || !filmstrip || !mainImage) return;

    let activeCategory = "identity";
    let activeNumber = 5;
    let lastFocusedElement = null;

    function currentGroup() {
      return supplementaryFigures.filter(function (figure) {
        return figure.category === activeCategory;
      });
    }

    function currentFigure() {
      return supplementaryFigures.find(function (figure) {
        return figure.number === activeNumber;
      });
    }

    function renderFilmstrip() {
      filmstrip.textContent = "";
      currentGroup().forEach(function (figure) {
        const button = document.createElement("button");
        const thumbnail = document.createElement("img");
        const copy = document.createElement("span");
        const number = document.createElement("small");
        const name = document.createElement("strong");

        button.type = "button";
        button.className = "supp-thumb";
        button.dataset.suppFigure = String(figure.number);
        button.setAttribute("aria-label", "Show supplementary Figure " + figure.number + ": " + figure.title);
        button.classList.toggle("is-active", figure.number === activeNumber);
        button.setAttribute("aria-pressed", String(figure.number === activeNumber));

        thumbnail.src = figure.src;
        thumbnail.alt = "";
        thumbnail.loading = "lazy";
        number.textContent = "Figure " + figure.number;
        name.textContent = figure.title;
        copy.appendChild(number);
        copy.appendChild(name);
        button.appendChild(thumbnail);
        button.appendChild(copy);
        button.addEventListener("click", function () {
          selectFigure(figure.number);
        });
        filmstrip.appendChild(button);
      });
    }

    function selectFigure(number) {
      const figure = supplementaryFigures.find(function (item) {
        return item.number === Number(number);
      });
      if (!figure) return;

      activeCategory = figure.category;
      activeNumber = figure.number;
      const group = currentGroup();
      const index = group.findIndex(function (item) {
        return item.number === activeNumber;
      });

      categoryButtons.forEach(function (button) {
        const active = button.dataset.suppCategory === activeCategory;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-selected", String(active));
      });

      categoryLabel.textContent = figure.categoryLabel;
      counter.textContent = String(index + 1) + " / " + String(group.length);
      figureNumber.textContent = "Supplementary Figure " + figure.number;
      title.textContent = figure.title;
      figcaptionLabel.textContent = "Supplementary Figure " + figure.number + ". " + figure.title + ".";
      caption.textContent = figure.caption;
      tags.textContent = "";
      figure.tags.forEach(function (tag) {
        const element = document.createElement("span");
        element.textContent = tag;
        tags.appendChild(element);
      });
      swapImage(mainImage, figure.src, figure.alt);

      Array.from(filmstrip.querySelectorAll("[data-supp-figure]")).forEach(function (button) {
        const active = Number(button.dataset.suppFigure) === figure.number;
        button.classList.toggle("is-active", active);
        button.setAttribute("aria-pressed", String(active));
      });

      if (!lightbox.hidden) {
        lightboxImage.src = figure.src;
        lightboxImage.alt = figure.alt;
        lightboxNumber.textContent = "Supplementary Figure " + figure.number;
        lightboxTitle.textContent = figure.title;
      }

      const nearby = [group[(index + 1) % group.length].src, group[(index - 1 + group.length) % group.length].src];
      preloadImages(nearby);
    }

    function selectCategory(category, preferredNumber) {
      const group = supplementaryFigures.filter(function (figure) {
        return figure.category === category;
      });
      if (group.length === 0) return;
      activeCategory = category;
      activeNumber = group.some(function (figure) { return figure.number === Number(preferredNumber); })
        ? Number(preferredNumber)
        : group[0].number;
      renderFilmstrip();
      selectFigure(activeNumber);
    }

    function stepFigure(direction) {
      const group = currentGroup();
      const index = group.findIndex(function (figure) {
        return figure.number === activeNumber;
      });
      const nextIndex = (index + direction + group.length) % group.length;
      selectFigure(group[nextIndex].number);
    }

    function openLightbox() {
      const figure = currentFigure();
      if (!figure || !lightbox) return;
      lastFocusedElement = document.activeElement;
      lightboxImage.src = figure.src;
      lightboxImage.alt = figure.alt;
      lightboxNumber.textContent = "Supplementary Figure " + figure.number;
      lightboxTitle.textContent = figure.title;
      lightbox.hidden = false;
      document.body.classList.add("lightbox-open");
      lightboxClose.focus();
    }

    function closeLightbox() {
      if (!lightbox || lightbox.hidden) return;
      lightbox.hidden = true;
      document.body.classList.remove("lightbox-open");
      if (lastFocusedElement) lastFocusedElement.focus();
    }

    categoryButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        selectCategory(button.dataset.suppCategory);
      });
    });

    jumpButtons.forEach(function (button) {
      button.addEventListener("click", function () {
        selectCategory(button.dataset.suppJump, button.dataset.suppFigure);
        section.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });

    previousButton.addEventListener("click", function () { stepFigure(-1); });
    nextButton.addEventListener("click", function () { stepFigure(1); });
    openButton.addEventListener("click", openLightbox);
    imageButton.addEventListener("click", openLightbox);
    lightboxClose.addEventListener("click", closeLightbox);
    lightboxBackdrop.addEventListener("click", closeLightbox);

    document.addEventListener("keydown", function (event) {
      if (!lightbox || lightbox.hidden) return;
      if (event.key === "Escape") closeLightbox();
      if (event.key === "ArrowLeft") stepFigure(-1);
      if (event.key === "ArrowRight") stepFigure(1);
    });

    renderFilmstrip();
    selectFigure(activeNumber);
  }

  function setupBibtexCopy() {
    const button = document.getElementById("copy-bibtex");
    const code = document.getElementById("bibtex-code");
    if (!button || !code) return;

    function showCopied() {
      const label = button.querySelector("span");
      button.classList.add("is-copied");
      label.textContent = "Copied";
      window.setTimeout(function () {
        button.classList.remove("is-copied");
        label.textContent = "Copy";
      }, 1800);
    }

    button.addEventListener("click", function () {
      const citation = code.textContent.trim();
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(citation).then(showCopied).catch(function () {
          fallbackCopy(citation, showCopied);
        });
      } else {
        fallbackCopy(citation, showCopied);
      }
    });
  }

  function fallbackCopy(text, onSuccess) {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.setAttribute("readonly", "");
    textArea.style.position = "fixed";
    textArea.style.opacity = "0";
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand("copy");
      onSuccess();
    } finally {
      document.body.removeChild(textArea);
    }
  }

  function setupActiveNavigation() {
    const links = Array.from(document.querySelectorAll(".nav-links a"));
    if (!("IntersectionObserver" in window) || links.length === 0) return;

    const sections = links
      .map(function (link) {
        return document.querySelector(link.getAttribute("href"));
      })
      .filter(Boolean);

    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (link) {
          link.classList.toggle("is-active", link.getAttribute("href") === "#" + entry.target.id);
        });
      });
    }, { rootMargin: "-30% 0px -60%", threshold: 0 });

    sections.forEach(function (section) {
      observer.observe(section);
    });
  }

  function setupSmoothAnchors() {
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduceMotion) return;

    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
      link.addEventListener("click", function (event) {
        const target = document.querySelector(link.getAttribute("href"));
        if (!target) return;
        event.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        window.history.replaceState(null, "", link.getAttribute("href"));
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupRatioControl();
    setupSelectiveExamples();
    setupOrderComparison();
    setupSupplementExplorer();
    setupBibtexCopy();
    setupActiveNavigation();
    setupSmoothAnchors();
  });
})();
