import "./About.css";
import about_img from "../../../assets/img4.png";

const About = () => {
  return (
    <div className="about">
      <div className="about-left">
        <img src={about_img} alt="" className="about-img" />
      </div>
      <div className="about-right">
        <h3>TUNGKOL SA APLIKASYON</h3>
        <h2>Baybay Salita Aplikasyon</h2>
        <p>
          Maligayang pagdating sa BAYBAYSALITA, isang makabago at dynamic na
          plataporma sa pag-aaral na naglalayong pahusayin ang kakayahan sa
          pag-unawa sa pagbasa ng mga estudyanteng Grade 1 sa Taytay Elementary
          School. Ang aming misyon ay pagyamanin ang pagmamahal sa pagbabasa at
          tiyakin na ang bawat batang mag-aaral ay may matibay na pundasyon para
          sa tagumpay sa edukasyon.
        </p>
        <p>
          Sa BAYBAYSALITA, kinikilala namin ang mahalagang papel ng maagang
          kakayahan sa pagbabasa sa paglalakbay ng isang bata sa edukasyon. Ang
          aming plataporma ay idinisenyo para sa natatanging pangangailangan ng
          mga batang Grade 1 sa Pilipinas, na nagbibigay ng nakakaengganyo at
          makabuluhang nilalaman na kaakibat ng kanilang pang-araw-araw na
          karanasan. Sa pamamagitan ng mga interaktibong aralin, nakakaakit na
          mga kwento, at mga personalisadong landas ng pagkatuto, layunin naming
          gawing masaya at kapaki-pakinabang ang pagbabasa para sa bawat bata.
        </p>
        <p>
          Ang aming komprehensibong approach ay pinagsasama ang pinakabagong
          pananaliksik sa edukasyon at praktikal na mga estratehiya sa pagtuturo
          upang suportahan ang mga estudyante sa pag-develop ng malakas na
          kakayahan sa pag-unawa sa pagbasa. Nag-aalok kami ng ibat ibang
          mapagkukunan, kabilang ang mga digital na libro, interaktibong
          ehersisyo, at mga tool sa pagtatasa, na lahat ay iniayon sa
          pangangailangan ng mga batang mambabasa. Sa suporta ng mga dedikadong
          guro at magulang, ang BAYBAYSALITA ay nagsusumikap na lumikha ng isang
          sumusuporta at nakakapagpayamang kapaligiran kung saan ang mga
          estudyante ay maaaring umunlad.
        </p>
      </div>
    </div>
  );
};

export default About;
